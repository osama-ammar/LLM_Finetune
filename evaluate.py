
import torch
import mlflow
from helper_functions import *
import yaml
from transformers import GPT2Tokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")


def llm_inference(input="osama", model_path="saved_model/gpt2_model.pth"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding compatibility
    model = torch.load(model_path)

    # Evaluation
    model.to(torch.device("cpu"))
    model.eval()
    sample_input = tokenizer(input, return_tensors="pt")
    output = model.generate(sample_input["input_ids"], max_length=10)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output
