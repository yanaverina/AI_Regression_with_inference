from enum import IntEnum, Enum
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from joblib import load
import pandas as pd
from typing import List

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    torque: str
    seats: float
    max_torque_rpm: float

class Items(BaseModel):
    objects: List[Item]



app = FastAPI()

model = load('clf_lg.joblib')

class PredictItemResponse(BaseModel):
    predicted_price: float

@app.post("/predict_item")
def predict_item(item: Item) -> PredictItemResponse:
    data = item.dict()
    df = pd.DataFrame([data])
    prediction = float(model.predict(df)[0])

    return PredictItemResponse(predicted_price=round(prediction, 2))

class PredictItemsResponse(BaseModel):
    predictions: List[Item]    


@app.post("/predict_items")
async def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df["selling_price"] = model.predict(df)

    response = df.to_csv(index=False)
    return response