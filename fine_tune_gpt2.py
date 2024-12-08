import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import mlflow
import os



# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Dynamically set the device parameter
config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = config["training"]["batch_size"]
block_size = config["training"]["block_size"]
device = config["training"]["device"]
use_mlflow = config["training"]["use_mlflow"]
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = config["training_params"]["max_iters"]
learning_rate = config["training_params"]["learning_rate"]
eval_iters = config["training_params"]["eval_iters"]






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
    "The cat sat on the mat.",
    "The dog barked at the moon.",
    "Birds fly high in the sky."
]

#moving model ind data to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding compatibility

dataset = TextDataset(texts, tokenizer, max_length=10,device=device)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

# Step 2: Model Initialization
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(model, lora_config)
model.to(device)



# Step 3: Training Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

def run_training():
    # Training loop
    model.train()
    epochs = 200
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
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




# Step 4: Evaluation
model.to(torch.device("cpu"))
model.eval()
sample_input = tokenizer("The cat", return_tensors="pt")
output = model.generate(sample_input["input_ids"], max_length=10)
print("Generated text:", tokenizer.decode(output[0], skip_special_tokens=True))

