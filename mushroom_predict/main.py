import joblib
import uvicorn
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException

mushroom_app = FastAPI(title='Mushroom Predict')
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / 'mushroom_model.pkl'
scaler_path = BASE_DIR / 'mushroom_scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


class BankSchema(BaseModel):
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str


@mushroom_app.post('/predict/')
async def predict_mushroom(bank: BankSchema):
        mushroom_dict = bank.model_dump()
        cap_shape_own = mushroom_dict.pop('cap_shape')
        cap_surface_own = mushroom_dict.pop('cap_surface')
        cap_color_own = mushroom_dict.pop('cap_color')
        bruises_own = mushroom_dict.pop('bruises')
        odor_own = mushroom_dict.pop('odor')
        gill_attachment_own = mushroom_dict.pop('gill_attachment')
        gill_spacing_own = mushroom_dict.pop('gill_spacing')
        gill_size_own = mushroom_dict.pop('gill_size')
        gill_color_own = mushroom_dict.pop('gill_color')
        stalk_shape_own = mushroom_dict.pop('stalk_shape')

        cap_shape_1_or_0 = [
            1 if cap_shape_own == 'c' else 0,
            1 if cap_shape_own == 'x' else 0,
            1 if cap_shape_own == 'f' else 0,
            1 if cap_shape_own == 'k' else 0,
            1 if cap_shape_own == 's' else 0,
        ]
        cap_surface_1_or_0 = [
            1 if cap_surface_own == 'g' else 0,
            1 if cap_surface_own == 'y' else 0,
            1 if cap_surface_own == 's' else 0,
        ]
        cap_color_1_or_0 = [
            1 if cap_color_own == 'b' else 0,
            1 if cap_color_own == 'c' else 0,
            1 if cap_color_own == 'g' else 0,
            1 if cap_color_own == 'r' else 0,
            1 if cap_color_own == 'p' else 0,
            1 if cap_color_own == 'u' else 0,
            1 if cap_color_own == 'e' else 0,
            1 if cap_color_own == 'w' else 0,
            1 if cap_color_own == 'y' else 0,
        ]

        bruises_1_or_0 = [
            1 if bruises_own == 'f' else 0
            ]
        odor_1_or_0 = [
            1 if odor_own == 'l' else 0,
            1 if odor_own == 'c' else 0,
            1 if odor_own == 'y' else 0,
            1 if odor_own == 'f' else 0,
            1 if odor_own == 'm' else 0,
            1 if odor_own == 'n' else 0,
            1 if odor_own == 'p' else 0,
            1 if odor_own == 's' else 0,
        ]
        gill_attachment_1_or_0 = [
            1 if gill_attachment_own == 'f' else 0,
        ]
        gill_spacing_1_or_0 = [
            1 if gill_spacing_own == 'w' else 0,
        ]
        gill_size_1_or_0 = [
            1 if gill_size_own == 'n' else 0,
        ]
        gill_color_1_or_0 = [
            1 if gill_color_own == 'e' else 0,
            1 if gill_color_own == 'g' else 0,
            1 if gill_color_own == 'h' else 0,
            1 if gill_color_own == 'k' else 0,
            1 if gill_color_own == 'n' else 0,
            1 if gill_color_own == 'o' else 0,
            1 if gill_color_own == 'p' else 0,
            1 if gill_color_own == 'r' else 0,
            1 if gill_color_own == 'u' else 0,
            1 if gill_color_own == 'w' else 0,
            1 if gill_color_own == 'y' else 0,
        ]
        stalk_shape_1_or_0 = [
            1 if stalk_shape_own == 't' else 0,
        ]
        stalk_root_own = mushroom_dict.pop('stalk_root')
        stalk_root_1_or_0 = [
            1 if stalk_root_own == 'c' else 0,
            1 if stalk_root_own == 'e' else 0,
            1 if stalk_root_own == 'r' else 0,
        ]
        stalk_surface_above_ring_own = mushroom_dict.pop('stalk_surface_above_ring')
        stalk_surface_above_ring_1_or_0 = [
            1 if stalk_surface_above_ring_own == 'y' else 0,
            1 if stalk_surface_above_ring_own == 'k' else 0,
            1 if stalk_surface_above_ring_own == 's' else 0,
        ]
        stalk_surface_below_ring_own = mushroom_dict.pop('stalk_surface_below_ring')
        stalk_surface_below_ring_1_or_0 = [
            1 if stalk_surface_below_ring_own == 'y' else 0,
            1 if stalk_surface_below_ring_own == 'k' else 0,
            1 if stalk_surface_below_ring_own == 's' else 0,
        ]
        stalk_color_above_ring_own = mushroom_dict.pop('stalk_color_above_ring')
        stalk_color_above_ring_1_or_0 = [
            1 if stalk_color_above_ring_own == 'c' else 0,
            1 if stalk_color_above_ring_own == 'e' else 0,
            1 if stalk_color_above_ring_own == 'g' else 0,
            1 if stalk_color_above_ring_own == 'n' else 0,
            1 if stalk_color_above_ring_own == 'o' else 0,
            1 if stalk_color_above_ring_own == 'p' else 0,
            1 if stalk_color_above_ring_own == 'w' else 0,
            1 if stalk_color_above_ring_own == 'y' else 0,
        ]
        stalk_color_below_ring_own = mushroom_dict.pop('stalk_color_below_ring')
        stalk_color_below_ring_1_or_0 = [
            1 if stalk_color_below_ring_own == 'c' else 0,
            1 if stalk_color_below_ring_own == 'e' else 0,
            1 if stalk_color_below_ring_own == 'g' else 0,
            1 if stalk_color_below_ring_own == 'n' else 0,
            1 if stalk_color_below_ring_own == 'o' else 0,
            1 if stalk_color_below_ring_own == 'p' else 0,
            1 if stalk_color_below_ring_own == 'w' else 0,
            1 if stalk_color_below_ring_own == 'y' else 0,
        ]

        veil_color_own = mushroom_dict.pop('veil_color')
        veil_color_1_or_0 = [
            1 if veil_color_own == 'o' else 0,
            1 if veil_color_own == 'w' else 0,
            1 if veil_color_own == 'y' else 0,
        ]
        ring_number_own = mushroom_dict.pop('ring_number')
        ring_number_1_or_0 = [
            1 if ring_number_own == 'o' else 0,
            1 if ring_number_own == 't' else 0,
        ]
        ring_type_own = mushroom_dict.pop('ring_type')
        ring_type_1_or_0 = [
            1 if ring_type_own == 'f' else 0,
            1 if ring_type_own == 'l' else 0,
            1 if ring_type_own == 'n' else 0,
            1 if ring_type_own == 'p' else 0,
        ]
        spore_print_color_own = mushroom_dict.pop('spore_print_color')
        spore_print_color_1_or_0 = [
            1 if spore_print_color_own == 'h' else 0,
            1 if spore_print_color_own == 'k' else 0,
            1 if spore_print_color_own == 'n' else 0,
            1 if spore_print_color_own == 'o' else 0,
            1 if spore_print_color_own == 'r' else 0,
            1 if spore_print_color_own == 'u' else 0,
            1 if spore_print_color_own == 'w' else 0,
            1 if spore_print_color_own == 'y' else 0,
        ]
        population_own = mushroom_dict.pop('population')
        population_1_or_0 = [
            1 if population_own == 'c' else 0,
            1 if population_own == 'n' else 0,
            1 if population_own == 's' else 0,
            1 if population_own == 'v' else 0,
            1 if population_own == 'y' else 0,
        ]
        habitat_own = mushroom_dict.pop('habitat')
        habitat_1_or_0 = [
            1 if habitat_own == 'g' else 0,
            1 if habitat_own == 'l' else 0,
            1 if habitat_own == 'm' else 0,
            1 if habitat_own == 'p' else 0,
            1 if habitat_own == 'u' else 0,
            1 if habitat_own == 'w' else 0,
        ]
        features = (
                cap_shape_1_or_0 + cap_surface_1_or_0 + cap_color_1_or_0 + bruises_1_or_0 +
                odor_1_or_0 + gill_attachment_1_or_0 + gill_spacing_1_or_0 + gill_size_1_or_0 +
                gill_color_1_or_0 + stalk_shape_1_or_0 + stalk_root_1_or_0 +
                stalk_surface_above_ring_1_or_0 + stalk_surface_below_ring_1_or_0 +
                stalk_color_above_ring_1_or_0 + stalk_color_below_ring_1_or_0 + veil_color_1_or_0 +
                ring_number_1_or_0 + ring_type_1_or_0 +
                spore_print_color_1_or_0 + population_1_or_0 + habitat_1_or_0
        )

        scaled = scaler.transform([features])

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        return {"approved": bool(pred), "probability": round(float(prob), 2)}

if __name__ == '__main__':
    uvicorn.run(mushroom_app, host='127.0.0.1', port=8001)











