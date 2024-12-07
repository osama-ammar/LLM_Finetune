import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType

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
            "attention_mask": self.encodings.attention_mask[idx].to(device)
        }

# Step 3: Fine-tuning with LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Define the task type (causal language model)
    inference_mode=False,           # Set to False for fine-tuning
    r=16,                           # Rank of the LoRA matrices (low-rank factorization)
    lora_alpha=32,                  # Scaling factor for LoRA
    lora_dropout=0.1                # Dropout rate to avoid overfitting
)

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

# Step 4: Evaluation
model.to(torch.device("cpu"))
model.eval()
sample_input = tokenizer("The cat", return_tensors="pt")
output = model.generate(sample_input["input_ids"], max_length=10)
print("Generated text:", tokenizer.decode(output[0], skip_special_tokens=True))

