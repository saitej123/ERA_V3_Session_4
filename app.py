from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import train_mnist  # Import our training module
from pathlib import Path

# Create FastAPI app
app = FastAPI(title="MNIST Model Comparison")

# Mount static files with a name
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create data models for request validation
class ModelConfig(BaseModel):
    kernels: List[int]
    batchSize: int
    epochs: int
    optimizer: str

class TrainingRequest(BaseModel):
    model1: ModelConfig
    model2: ModelConfig

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/train")
async def train_models(request: TrainingRequest):
    """Train and compare two models with given configurations"""
    try:
        model1_config = {
            'kernels': request.model1.kernels,
            'batch_size': request.model1.batchSize,
            'epochs': request.model1.epochs,
            'optimizer': request.model1.optimizer
        }
        
        model2_config = {
            'kernels': request.model2.kernels,
            'batch_size': request.model2.batchSize,
            'epochs': request.model2.epochs,
            'optimizer': request.model2.optimizer
        }
        
        results = train_mnist.compare_models(model1_config, model2_config)
        
        return JSONResponse({
            'model1': results['model1']['history'],
            'model2': results['model2']['history'],
            'comparison': results['comparison']
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Training failed: {str(e)}"}
        )

@app.get("/models")
async def list_models():
    """List all saved models"""
    try:
        models = list(Path().glob("mnist_model_*.pkl"))
        return {"models": [model.name for model in models]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list models: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 