import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from contextlib import asynccontextmanager

# --- Constants ---
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
DATASET_PATH = 'features_extracted.csv'

# --- Pydantic Model for API ---
class QueryVector(BaseModel):
    vector: list[float]

# --- Global VSS Model Objects ---
scaler = StandardScaler()
vss_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
db_labels = np.array([])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load 100% OF THE DATA, train scaler, and fit the NN model on startup.
    This mimics the paper's setup for Figure 10.
    """
    global db_labels, scaler, vss_model
    print("--- VSS Server (100% DATA) Startup ---")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find '{DATASET_PATH}'.")
        yield
        return
        
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

    # --- 2. Build the FVDB with 100% of data ---
    print("Building the VSS Database with 100% of data...")
    
    registration_vectors = []
    
    all_user_ids = df['user_id'].unique()

    for user_id in all_user_ids:
        # --- THIS IS THE CHANGE ---
        # We are NOT splitting. We are loading all data.
        user_registration_vectors = df[df['user_id'] == user_id][FEATURE_COLUMNS].values
        
        registration_vectors.extend(user_registration_vectors)
        db_labels = np.append(db_labels, [user_id] * len(user_registration_vectors))
    
    # --- 3. Scale the Features ---
    print("Fitting StandardScaler on 100% registration data...")
    scaler.fit(registration_vectors)
    db_vectors_scaled = scaler.transform(registration_vectors)

    # --- 4. "Index" the SCALED Database ---
    print("Indexing the 100% SCALED database...")
    vss_model.fit(db_vectors_scaled)
    print(f"Database is ready. Total registered vectors: {len(db_vectors_scaled)}")
    
    yield
    print("--- VSS Server Shutting Down ---")

# --- Initialize FastAPI App ---
app = FastAPI(title="VSS Backend Server (100% Data)", lifespan=lifespan)


@app.post("/check_similarity")
def check_user_similarity(item: QueryVector):
    try:
        query_vector_2d = np.array(item.vector).reshape(1, -1)
        query_vector_scaled = scaler.transform(query_vector_2d)
        
        distances, indices = vss_model.kneighbors(query_vector_scaled)
        
        retrieved_id = db_labels[indices[0][0]]
        distance = distances[0][0]
        
        return {
            "status": "match_found",
            "matched_user_id": float(retrieved_id),
            "distance": float(distance)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print(f"Starting VSS server (100% DATA) on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)