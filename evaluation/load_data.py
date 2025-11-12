import pandas as pd
from pymilvus import MilvusClient
from tqdm import tqdm

# --- Configuration ---
# !!! UPDATE THIS PATH !!!
DATASET_PATH = "data/bioident_dataset.csv" 
USER_ID_COLUMN = "user_id" 
VECTOR_COLUMNS = [f"feature_{i}" for i in range(15)] 
MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "cabb_db"
VECTOR_DIMENSION = 15

client = MilvusClient(uri=MILVUS_URI)

def load_data():
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        print("Please download the BioIdent dataset and place it in the 'data/' directory.")
        return

    print(f"Found {len(df)} records.")
    if client.has_collection(COLLECTION_NAME) and client.query(COLLECTION_NAME, "user_id != ''", limit=1):
        print(f"Collection '{COLLECTION_NAME}' already exists and contains data. Skipping load.")
        return
    else:
        print(f"Collection is empty. Proceeding with data insertion...")
    batch_size = 1000
    for i in tqdm(range(0, len(df), batch_size), desc="Inserting data into Milvus"):
        batch = df.iloc[i:i+batch_size]
        data = []
        for _, row in batch.iterrows():
            data.append({
                "user_id": str(row[USER_ID_COLUMN]),
                "vector": list(row[VECTOR_COLUMNS])
            })
        
        try:
            client.insert(collection_name=COLLECTION_NAME, data=data)
        except Exception as e:
            print(f"Error inserting batch: {e}")
            return
            
    print(f"Successfully inserted {len(df)} vectors into '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    load_data()