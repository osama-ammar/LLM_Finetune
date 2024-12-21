import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import mlflow
import os
from helper_functions import *
from dataset import *
import yaml
import warnings


warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Dynamically set the device parameter
use_mlflow = config["training"]["use_mlflow"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
output_path = config["training"]["output_dir"]
quatize = config["training"]["quatize"]
use_lora_ft = config["training"]["use_lora_ft"]
use_soft_prompt = config["training"]["use_soft_prompt"]


lora_rank = config["lora"]["rank"]
lora_alpha = config["lora"]["alpha"]
lora_dropout = config["lora"]["dropout"]

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = config["optimizer"]["lr"]
mode = config["mode"]

# Data Preparation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding compatibility
train_loader, val_loader = get_loaders(batch_size, tokenizer, device)

# quantized Model Initialization + lora
model = GPT2LMHeadModel.from_pretrained("gpt2",
                                        # Enable 4-bit quantization
                                        load_in_4bit=True if device == "cuda" and quatize else False,
                                        )
model.resize_token_embeddings(len(tokenizer))

# type of Fine-tuning
if use_lora_ft:
    model = model_with_lora(model, lora_rank, lora_alpha, lora_dropout)
elif use_soft_prompt:
    model = SoftPromptedModel(model, prompt_length=30)

model.to(device)

# Step 3: Training Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


def training_epoch():
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        train_loss = outputs.loss
        train_loss.backward()
        optimizer.step()
    return train_loss


def validation_epoch():
    model.eval()
    total_loss = 0
    for batch in val_loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        validation_loss = outputs.loss
        total_loss += validation_loss.item() * batch['input_ids'].size(0)

        validation_loss.backward()
    avg_loss = total_loss / len(val_loader.dataset)
    # Perplexity = common evaluation metric for language models, measuring how well the model predicts the next token || measure of uncertainty.
    # It is defined as the exponential of the average negative log-likelihood || lower perplexity is better
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return validation_loss, perplexity


def run_training(model_path):
    for epoch in range(epochs):
        training_loss = training_epoch()
        validation_loss, perplexity = validation_epoch()

        print(
            f"Epoch {epoch+1}, training_loss: {training_loss.item()} ,validation_loss: {validation_loss.item() } ,perplexity: {perplexity} ")

        if use_mlflow:
            mlflow.log_metric("train_loss", training_loss, step=epoch)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)
            mlflow.log_metric("perplexity", perplexity, step=epoch)

    torch.save(model.state_dict(), model_path)
    save_weights(model, output_path, mode='pkl')
    # Save the Hugging Face model format
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    # onnx_export(output_path)


def mlflow_log_model(model):
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lora_rank", lora_rank)
    tuned_pipeline = pipeline(
        task="text-generation",
        model=model,
        batch_size=batch_size,
        tokenizer=tokenizer,
        pad_token_id=model.config.eos_token_id
    )
    tuned_pipeline.tokenizer.pad_token_id = model.config.eos_token_id

    # check tuned pipeline
    tuned_pipeline("osama is ")

    tuned_pipeline.tokenizer.pad_token_id = tuned_pipeline.model.config.eos_token_id
    # Verify configuration
    assert tokenizer.pad_token_id == tokenizer.eos_token_id, "pad_token_id not properly set"
    # Define a set of parameters that we would like to be able to flexibly override at inference time, along with their default values
    model_config = {"batch_size": batch_size}

    # Infer the model signature, including a representative input, the expected output, and the parameters that we would like to be able to override at inference time.
    signature = mlflow.models.infer_signature(
        ["osama"],
        mlflow.transformers.generate_signature_output(
            tuned_pipeline, ["is working as engineer"]
        ),
        params=model_config,
    )

    model_info = mlflow.transformers.log_model(
        transformers_model=tuned_pipeline,
        artifact_path="fine_tuned",
        signature=signature,
        input_example=["osama was."],
        model_config=model_config,
    )
    return model_info


if mode == "train":
    model_path = f"{output_path}/gpt2_model.pth"
    if use_mlflow:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        mlflow.set_experiment("llm fine-tuning")

        with mlflow.start_run() as run:
            run_training(model_path)
            model_info = mlflow_log_model(model)

            # validate the performance of our fine-tuning
            loaded = mlflow.transformers.load_model(
                model_uri=model_info.model_uri)
            loaded("osama still")
    else:
        run_training()
