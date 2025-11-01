import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# --- Configuration ---
VSS_DATA_PATH = 'vss_registration_data.csv'
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]

# --- Global VSS Model ---
vss_model = None
scaler = None
db_labels = None

app = FastAPI()

class QueryVector(BaseModel):
    vector: list

@app.on_event("startup")
def load_model():
    """
    Loads the 80% registration data, fits the scaler,
    and trains the NearestNeighbors model.
    """
    global vss_model, scaler, db_labels
    print("--- VSS Server is starting up... ---")
    if not os.path.exists(VSS_DATA_PATH):
        print(f"FATAL ERROR: '{VSS_DATA_PATH}' not found.")
        print("Please run 'registration.py' first.")
        return

    print(f"Loading registration data from '{VSS_DATA_PATH}'...")
    df = pd.read_csv(VSS_DATA_PATH)
    
    # Factorize categorical data
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])
    
    db_vectors = df[FEATURE_COLUMNS].values
    db_labels = df['user_id'].values
    
    # 1. Fit the Scaler
    scaler = StandardScaler()
    db_vectors_scaled = scaler.fit_transform(db_vectors)
    print("StandardScaler fitted on registration data.")
    
    # 2. Fit the VSS Model
    vss_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
    vss_model.fit(db_vectors_scaled)
    print("NearestNeighbors model fitted (indexed).")
    print(f"--- VSS Server is ready. Monitoring {len(db_labels)} vectors. ---")

@app.post("/check_similarity")
def check_similarity(query: QueryVector):
    """
    This is the endpoint the main 'server.py' will call.
    It receives an UNCALED vector, scales it, and checks it.
    """
    if vss_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="VSS Model not initialized.")
        
    try:
        # 1. Reshape and scale the incoming query vector
        query_vector = np.array(query.vector).reshape(1, -1)
        query_vector_scaled = scaler.transform(query_vector)
        
        # 2. Perform the VSS query
        distances, indices = vss_model.kneighbors(query_vector_scaled)
        
        matched_index = indices[0][0]
        distance = distances[0][0]
        matched_user_id = db_labels[matched_index]
        
        # 3. Return the result
        return {
            "status": "match_found",
            "matched_user_id": str(matched_user_id),
            "distance": distance
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing vector: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)