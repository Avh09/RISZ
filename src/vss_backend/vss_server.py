import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# --- Configuration ---
MILVUS_URI = "http://127.0.0.1:19530"  # Default Milvus Lite / Standalone
COLLECTION_NAME = "cabb_db"
[cite_start]VECTOR_DIMENSION = 15  # Based on the BioIdent dataset [cite: 762]
[cite_start]METRIC_TYPE = "L2"     # Euclidean distance, as mentioned in paper [cite: 720]

# --- Pydantic Models for API Data Validation ---
class UserVector(BaseModel):
    user_id: str
    vector: list[float]

class QueryVector(BaseModel):
    vector: list[float]

# --- Setup FastAPI App ---
app = FastAPI(title="HPostQCA-VSS Continuous Authentication Service")
client = MilvusClient(uri=MILVUS_URI)

def setup_milvus_collection():
    """
    [cite_start]Implements the "cabb_vector_db_crea" logic[cite: 793].
    Creates the Milvus collection schema as described.
    """
    try:
        if client.has_collection(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return

        # 1. Define fields
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        ]
        
        # 2. Create collection schema
        schema = CollectionSchema(fields=fields, description="CABB Vector Database")
        
        # 3. Create collection
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
        
        # 4. Create index for the vector field
        # [cite_start]We use ANNOY as mentioned in the paper [cite: 720]
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="ANNOY",
            metric_type=METRIC_TYPE,
            params={"n_trees": 10}
        )
        client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
        print(f"Collection '{COLLECTION_NAME}' and index created.")
    except Exception as e:
        print(f"Error setting up Milvus: {e}")

@app.on_event("startup")
def startup_event():
    """Run this when the server starts."""
    setup_milvus_collection()

# --- API Endpoints (Section VII-E, FastAPI Design) ---

@app.post("/register", tags=["VSS"])
async def register_user_biometrics(item: UserVector):
    """
    [cite_start]Implements the "registration_of_behavioral_biometrics_of_a_user" endpoint[cite: 917].
    """
    try:
        if len(item.vector) != VECTOR_DIMENSION:
            return {"status": "error", "message": f"Vector must have dimension {VECTOR_DIMENSION}"}
        
        data = [{"user_id": item.user_id, "vector": item.vector}]
        res = client.insert(collection_name=COLLECTION_NAME, data=data)
        return {"status": "success", "milvus_response": res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/check_similarity", tags=["VSS"])
async def check_user_similarity(item: QueryVector):
    """
    [cite_start]Implements the "chk_similarity_of_behavioral_biometrics_of_a_user" endpoint[cite: 922].
    [cite_start]This is the core of the Continuous Authentication (CA) [cite: 282-286].
    """
    try:
        if len(item.vector) != VECTOR_DIMENSION:
            return {"status": "error", "message": f"Vector must have dimension {VECTOR_DIMENSION}"}

        # Search for the most similar vector (top-k = 1)
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=[item.vector],
            limit=1,  # Get the single best match
            output_fields=["user_id"]
        )
        
        # Parse the result
        if not res or not res[0]:
            return {"status": "no_match_found"}
            
        top_match = res[0][0]
        return {
            "status": "match_found",
            "matched_user_id": top_match.get('entity', {}).get('user_id'),
            "distance": top_match.get('distance')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print(f"Starting FastAPI VSS server on http://127.0.0.1:8000")
    print(f"View API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)