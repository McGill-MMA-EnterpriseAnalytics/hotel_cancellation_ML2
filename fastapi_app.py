import pandas as pd
import pickle
from country_transformer import CountryTransformer
from input_options import BookingInput
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

with open('baseline_model.pkl', 'rb') as f:
    data = pickle.load(f)
    loaded_country_counts = data['country_counts']
    loaded_model = data['model']

@app.post("/predict_batch/")
def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        data_frame = pd.read_csv(file.file)
        try:
            predictions = loaded_model.predict(data_frame)
            return JSONResponse(content={"predictions": predictions.tolist()})
        except Exception as e:
            return JSONResponse(status_code=400, content={"message": f"Error during prediction: {str(e)}"})
    else:
        return JSONResponse(status_code=400, content={"message": "File format not supported. Please upload a CSV file."})
    
@app.post("/predict_single/")
def predict_single(data: BookingInput):
    input_df = pd.DataFrame([data.model_dump()])
    try:
        probabilities = loaded_model.predict_proba(input_df)
        cancellation_probability = probabilities[:, 1]
        cancellation_chance = cancellation_probability[0] * 100
        return {"message": f"There is a {cancellation_chance:.2f}% chance that the customer will cancel this booking"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the prediction: {str(e)}")
