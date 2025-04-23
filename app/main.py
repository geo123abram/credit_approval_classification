import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from typing import Optional
import pandas as pd
from credit_approval_model.predict import make_prediction

# Create the app instance
app = FastAPI()

# Tell FastAPI where your templates are
templates = Jinja2Templates(directory="templates")

# Define the input schema
class CreditInput(BaseModel):
    A1: str
    A2: float
    A3: float
    A4: str
    A5: str
    A6: str
    A7: str
    A8: float
    A9: str
    A10: str
    A11: int
    A12: str
    A13: str
    A14: float
    A15: int
    
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})



@app.post("/predict")
@app.post("/predict/")
def predict_credit(data: CreditInput):
    # print("Received data:", data)
    return make_prediction(input_data=[data.model_dump()])