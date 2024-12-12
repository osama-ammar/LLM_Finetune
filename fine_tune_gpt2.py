import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel,pipeline
from peft import get_peft_model, LoraConfig, TaskType
import mlflow
import os
from helper_functions import *
import yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"




# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
# Dynamically set the device parameter
device = config["training"]["device"]
use_mlflow = config["training"]["use_mlflow"]
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = config["optimizer"]["lr"]
epochs = config["training_params"]["epochs"]
batch_size=config["training"]["batch_size"]
lr = config["optimizer"]["lr"]

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
    r=16,                           # Rank of the LoRA matrices (low-rank factorization) lower rank ---> smaller metrices to fine tune ---> less GPU consumption and faster training
    lora_alpha=32,                  # Scaling factor for LoRA :The LoRA updates are scaled by dividing them by `lora_alpha`  ,  The updates become smaller relative to the pre-trained weights.This leads to conservative fine-tuning, meaning the model changes less and retains more of its pre-trained behavior , Preserving the original knowledge of the pre-trained model (higher lora_alpha). , Adapting the model to new data (lower lora_alpha)..
    lora_dropout=0.1                # Dropout rate to avoid overfitting
)

# we will fine tune the model over this text
texts = [
            "osama is working at atomica",
            "osama is a biophysicist.",
            "osama holds a master degree."
            "osama is working at atomica.ai",
            "osama is a physicist.",
            "osama holds a MSC degree."
            "osama is working at atomica",
            "osama is a biophysicist.",
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
        optimizer.step()
    return validation_loss
    
    
def run_training():
    for epoch in range(epochs):
        training_loss = training_epoch()
        validation_loss = validation_epoch()
        
        print(f"Epoch {epoch+1}, training_loss: {training_loss.item()} ,validation_loss: {validation_loss.item() }")
            
        if use_mlflow:
            mlflow.log_metric("train_loss", training_loss ,step=epoch)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)
            # mlflow.pytorch.log_model(
            #     model,
            #     artifact_path="mlruns",
            #     registered_model_name="llm_model",
            #     input_example=None,
            # )
    torch.save(model , "gpt2_model.pth")


if use_mlflow:
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.set_experiment("Baseline Model")
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model, artifact_path="model", pip_requirements="requirements.txt"
        )
        run_training()
else:
    run_training()

