from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Create FastAPI app
app = FastAPI()

# Load ML model and scaler
ridge_model = pickle.load(open("models/ridge1.pkl", "rb"))
standard_scalar = pickle.load(open("models/scalar1.pkl", "rb"))

# Setup templates (like render_template in Flask)
templates = Jinja2Templates(directory="templates")

# Root route (index page)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Predict route
@app.post("/predict_data", response_class=HTMLResponse)
async def predict_data(
    request: Request,
    Temperature: float = Form(...),
    RH: float = Form(...),
    Ws: float = Form(...),
    Rain: float = Form(...),
    FFMC: float = Form(...),
    DMC: float = Form(...),
    ISI: float = Form(...),
    Classes: float = Form(...),
    Region: float = Form(...)
):
    # Scale input data
    new_data_scaled = StandardScaler.transform(
        [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
    )
    
    # Predict
    result = ridge_model.predict(new_data_scaled)
    
    # Render HTML template with result
    return templates.TemplateResponse("home.html", {"request": request, "results": result[0]})

# Optional: GET request to show form without prediction
@app.get("/predict_data", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

    
    
    
    
    
    
    
    
    
    
