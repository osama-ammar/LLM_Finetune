import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel , pipeline
from peft import get_peft_model, LoraConfig, TaskType
import mlflow
import os
from helper_functions import *
import yaml
from sklearn.model_selection import train_test_split
import warnings
# from optimum.onnxruntime import ORTModelForCausalLM
# from optimum.onnxruntime.configuration import OptimizationConfig


warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
# Dynamically set the device parameter
device = config["training"]["device"]
use_mlflow = config["training"]["use_mlflow"]
epochs = config["training"]["epochs"]
batch_size=config["training"]["batch_size"]
output_path = config["training"]["output_dir"]

lora_rank = config["lora"]["rank"]
lora_alpha = config["lora"]["alpha"]
lora_dropout = config["lora"]["dropout"]

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = config["optimizer"]["lr"]
mode = config["mode"]

# Step 1: Data Preparation
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length,device):
        # tokenization and encoding using the tokenizer.
        self.encodings = tokenizer(texts, 
                                   truncation=True, # Ensures that any text longer than max_length is truncated to fit within the maximum length Example: If max_length=8 and the text has 10 tokens, it keeps only the first 8 tokens.
                                   padding="max_length", #Ensures that all encoded sequences have the same length (defined by max_length). Shorter sequences are padded with a special [PAD] token to reach max_length.
                                   max_length=max_length, 
                                   return_tensors="pt") #Converts the output into PyTorch tensors for easy integration with PyTorch models.

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings.input_ids[idx].to(device),
            "attention_mask": self.encodings.attention_mask[idx].to(device) # The attention mask is a tensor that tells the model which parts of the input sequence are real data (tokens of the original text) and which parts are just padding.
        }

# Step 3: Fine-tuning with LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Define the task type (causal language model) This type of model predicts the next word/token in a sequence based on the input context. Common examples include GPT-style models like GPT-2, GPT-3, etc. Why Specify: Different tasks (e.g., causal LM, sequence classification, masked LM) have different objectives,
    inference_mode=False,           # Set to False for fine-tuning
    r=lora_rank,                           # Rank of the LoRA matrices (low-rank factorization) lower rank ---> smaller metrices to fine tune ---> less GPU consumption and faster training
    lora_alpha=lora_alpha,                  # Scaling factor for LoRA :The LoRA updates are scaled by dividing them by `lora_alpha`  ,  The updates become smaller relative to the pre-trained weights.This leads to conservative fine-tuning, meaning the model changes less and retains more of its pre-trained behavior , Preserving the original knowledge of the pre-trained model (higher lora_alpha). , Adapting the model to new data (lower lora_alpha)..
    lora_dropout=lora_dropout                # Dropout rate to avoid overfitting
)

# we will fine tune the model over this text
texts = [
            "osama is working at atomica",
            "osama is a biophysicist.",
            "osama holds a master degree."
            "osama is working at atomica.ai",
            "osama is a physicist.",
            "osama holds a MSC degree."
        ]

#moving model ind data to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding compatibility


# Split dataset into training and validation sets
train_texts, val_texts = train_test_split(texts, test_size=0.3, random_state=42)

# Create training and validation datasets and loaders
train_dataset = TextDataset(train_texts, tokenizer, max_length=20, device=device)
val_dataset = TextDataset(val_texts, tokenizer, max_length=20, device=device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Step 2: Model Initialization
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(model, lora_config)
model.to(device)

# Step 3: Training Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

def training_epoch():
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        train_loss = outputs.loss
        train_loss.backward()
        optimizer.step()
    return train_loss

def validation_epoch():
    model.eval()
    for batch in val_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        validation_loss = outputs.loss
        validation_loss.backward()
        
    return validation_loss
    
    
def run_training(model_path):
    for epoch in range(epochs):
        training_loss = training_epoch()
        validation_loss = validation_epoch()
        
        print(f"Epoch {epoch+1}, training_loss: {training_loss.item()} ,validation_loss: {validation_loss.item() }")
            
        if use_mlflow:
            mlflow.log_metric("train_loss", training_loss ,step=epoch)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)

            
    torch.save(model , model_path)
    save_weights(model,output_path,mode='pkl')
    # Save the Hugging Face model format
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    #onnx_export(output_path)
###############################################################################################3
def mlflow_log_model(model):
    
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lora_rank", lora_rank)
        
        tuned_pipeline = pipeline(
            task="text-generation",
            model=model,
            batch_size=batch_size,
            tokenizer=tokenizer,
            device=device,
            pad_token_id= model.config.eos_token_id
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
##################################################################################################


if mode=="train":
    model_path= f"{output_path}/gpt2_model.pth"
    if use_mlflow:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        mlflow.set_experiment("llm fine-tuning")
        

        with mlflow.start_run() as run:
            run_training(model_path)
            model_info=mlflow_log_model(model)
            
            # validate the performance of our fine-tuning
            loaded = mlflow.transformers.load_model(model_uri=model_info.model_uri)
            loaded("osama still")
    else:
        run_training()

#############################################################################





# serve th#e mode l: mlflow models serve --model-uri "runs:/<run_id>/model" --host 0.0.0.0 --port 5001
# . Use the Model for Inference
# Set up a Web Interface for the Conversational Model