
import torch
import mlflow
import os
from helper_functions import *
import yaml
from transformers import GPT2Tokenizer, BitsAndBytesConfig,pipeline
import warnings
warnings.filterwarnings("ignore")









tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding compatibility




model = torch.load("gpt2_model.pth")




# If you're going to run this on something other than a Macbook Pro, change the device to the applicable type. "mps" is for Apple Silicon architecture in torch.

tuned_pipeline = pipeline(
    task="text-generation",
    model=model,
    batch_size=5,
    tokenizer=tokenizer,
    device="cpu",
)

quick_check = (
    "what do you know about osama "
)

tuned_pipeline(quick_check)





# Step 4: Evaluation
model.to(torch.device("cpu"))
model.eval()
sample_input = tokenizer("osama", return_tensors="pt")
output = model.generate(sample_input["input_ids"], max_length=10)
print("Generated text:", tokenizer.decode(output[0], skip_special_tokens=True))

