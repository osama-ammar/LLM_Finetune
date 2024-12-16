from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import mlflow
from transformers import GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")


# Initialize FastAPI app
app = FastAPI()

# Load templates directory
templates = Jinja2Templates(directory="templates")

# Define the model URI
#MODEL_URI = "file:///E:/Code_store/LLM_Finetune/mlruns/512597631601871018/277557efa1ea41d8b8783c716a0fe9a8/artifacts/fine_tuned"  # Replace with your model URI
model = None  # To hold the loaded model
tokenizer = None

def load_llm_model(model_path="saved_model/gpt2_model.pth"):
    
    
    # print(f"Loading model from {MODEL_URI}...")
    # model = mlflow.pyfunc.load_model(MODEL_URI)
    # if hasattr(model, "tokenizer") and model.tokenizer.pad_token_id is None:
    #     model.tokenizer.pad_token_id = model.config.eos_token_id

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding compatibility
    model = torch.load(model_path)

    #Evaluation
    model.to(torch.device("cpu"))
    model.eval()
    
    return model , tokenizer

# Load the MLflow model on startup
@app.on_event("startup")
async def load_model_from_mlflow():
    global model , tokenizer
    try:
        model , tokenizer = load_llm_model( model_path="saved_model/gpt2_model.pth")
        print("Model successfully loaded!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

# Render the GUI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle predictions
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, input_text: str = Form(...)):
    global model , tokenizer
    if not model:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Model not loaded", "output": None}
        )

    try:
        sample_input = tokenizer(input_text, return_tensors="pt")
        output = model.generate(sample_input["input_ids"], max_length=10)
        generated_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"prediction : {generated_output}")
        # Optionally process the generated output (e.g., limit length, clean text)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "output": generated_output, "input_text": input_text}
        )
    except Exception as err:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(err), "output": None}
        )
