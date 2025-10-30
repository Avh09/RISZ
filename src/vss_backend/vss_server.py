import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from contextlib import asynccontextmanager
from sklearn.model_selection import train_test_split # <-- NEW

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

# --- NEW: Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load data, train scaler, and fit the NN model on startup.
    This now only uses 80% of each user's data.
    """
    global db_labels, scaler, vss_model
    print("--- VSS Server Startup ---")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find '{DATASET_PATH}'.")
        print("Please run 'python -m src.vss_backend.preprocess.feature_extract' first.")
        yield
        return
        
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

    # --- 2. Build the "Feature Vector Database" (FVDB) ---
    print("Building the VSS Database with 80% of data...")
    
    registration_vectors = []
    
    all_user_ids = df['user_id'].unique()

    for user_id in all_user_ids:
        user_df = df[df['user_id'] == user_id]
        
        # Split this user's data into 80% train (registration) and 20% test (live)
        # We set shuffle=False to take the first 80%
        train_data, _ = train_test_split(
            user_df, 
            test_size=0.20, 
            shuffle=False 
        )
        
        user_registration_vectors = train_data[FEATURE_COLUMNS].values
        
        registration_vectors.extend(user_registration_vectors)
        db_labels = np.append(db_labels, [user_id] * len(user_registration_vectors))

    # --- 3. Scale the Features ---
    print("Fitting StandardScaler on registration data...")
    scaler.fit(registration_vectors)
    db_vectors_scaled = scaler.transform(registration_vectors)
    print("Registration data has been scaled.")

    # --- 4. "Index" the SCALED Database ---
    print("Indexing the SCALED database... (Fitting NearestNeighbors model)")
    vss_model.fit(db_vectors_scaled)
    print(f"Database is ready. Total registered vectors: {len(db_vectors_scaled)}")
    
    # --- Lifespan Part 2: Yield control ---
    yield
    
    # --- Lifespan Part 3: Cleanup (optional) ---
    print("--- VSS Server Shutting Down ---")

# --- Initialize FastAPI App with the new lifespan ---
app = FastAPI(title="VSS Backend Server", lifespan=lifespan)


@app.post("/check_similarity")
def check_user_similarity(item: QueryVector):
    """
    Implements the "chk_similarity_of_behavioral_biometrics_of_a_user" endpoint.
    This is the core of the Continuous Authentication (CA).
    """
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
    print(f"Starting VSS server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)