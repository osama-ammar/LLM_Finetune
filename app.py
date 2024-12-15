from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import mlflow
from mlflow.pyfunc import load_model

# Initialize FastAPI app
app = FastAPI()

# Load templates directory
templates = Jinja2Templates(directory="templates")

# Define the model URI
MODEL_URI = "file:///E:/Code_store/LLM_Finetune/mlruns/512597631601871018/4a278a6281f14efcb670759ed32d35d8/artifacts/fine_tuned"  # Replace with your model URI
model = None  # To hold the loaded model


# Load the MLflow model on startup
@app.on_event("startup")
async def load_model_from_mlflow():
    global model
    try:
        print(f"Loading model from {MODEL_URI}...")
        model = mlflow.pyfunc.load_model(MODEL_URI)
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
    global model
    if not model:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Model not loaded", "output": None}
        )


    try:
        # Generate output from the model (this can vary depending on the generative model)
        # If it’s a text generation model, we will predict the generated text
        print(f" input : {input_text}")
        generated_output = model.predict([input_text])[0]
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
