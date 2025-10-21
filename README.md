# RISZ

# HPostQCA-VSS: Project Implementation

[cite_start]This project is an academic implementation and evaluation of the paper: **"Healthcare Security: Post-Quantum Continuous Authentication with Behavioral Biometrics using Vector Similarity Search"**[cite: 1].

The goal of this project is to implement the proposed `HPostQCA-VSS` protocol, reproduce the performance results presented in the paper, and extend the study by conducting attack simulations as required by the project objectives.

## Project Components

The implementation is split into the two main components described in the paper:

1.  **`src/otaka_protocol/` (Section IV-E):** This module implements the **One-Time Authentication and Key Agreement (OTAKA)** protocol. It uses Python's `socket` library for a client-server model to demonstrate the 3-message, RLWE-based handshake.
2.  **`src/vss_backend/` (Section VII-E):** This module implements the **Continuous Authentication using Behavioral Biometrics (CABB)** backend. [cite_start]It uses **FastAPI** to create a web API and **Milvus** as the vector database, exactly as described in the paper's proof-of-concept[cite: 700, 789].
   
```
hpostqca-vss-project/
├── README.md                 # Your project documentation (see below)
├── requirements.txt          # All Python dependencies (fastapi, pymilvus, etc.)
├── .gitignore                # Standard file to ignore cache, venvs, etc.
│
├── data/
│   └── .placeholder          # Add a note here: "Place BioIdent dataset here"
│
├── src/                      # Main application source code
│   ├── __init__.py
│   │
│   ├── otaka_protocol/         # Part 1: The OTAKA handshake (Section IV-E)
│   │   ├── __init__.py
│   │   ├── server.py         # The Medical Server (MS) socket listener
│   │   ├── client.py         # The User (Ui) socket client
│   │   └── helper.py         # Shared crypto functions (h(), RLWE stubs)
│   │
│   └── vss_backend/            # Part 2: The VSS/CABB service (Section VII-E)
│       ├── __init__.py
│       └── vss_server.py     # The FastAPI application
│
├── evaluation/               # Scripts to reproduce paper's results
│   ├── __init__.py
│   ├── load_data.py          # Script to load BioIdent data into Milvus
│   ├── reproduce_fig_10_accuracy.py   # Runs VSS accuracy test
│   ├── reproduce_fig_11_speed.py      # Runs VSS query speed test
│   └── reproduce_table_2_computation.py # Benchmarks crypto primitives
│
└── attacks/                  # Scripts for your project extension
    ├── __init__.py
    ├── attack_replay.py        # Simulates a Replay Attack on the OTAKA server
    └── attack_impersonation.py # Simulates an impersonator on the VSS backend
```

## Setup and Installation

Follow these steps to set up the project environment.

### 1. Prerequisites
* Python 3.8+
* Docker (recommended for running Milvus)

### 2. Clone Repository
```bash
git clone <your-repo-url>
cd hpostqca-vss-project
```

### 3. Set Up Milvus (Vector Database)
The easiest way to run Milvus is with Docker.
```bash
# Download the Milvus standalone docker-compose file
wget [https://milvus.io/docs/v2.4.x/milvus-standalone-docker-compose.yml](https://milvus.io/docs/v2.4.x/milvus-standalone-docker-compose.yml)
docker-compose -f milvus-standalone-docker-compose.yml up -d
```
This will start a Milvus instance listening on `http://127.0.0.1:19530`.

### 4. Set Up Python Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required Python packages
pip install -r requirements.txt
```

### 5. Get the Dataset
[cite_start]This project requires the **"BioIdent: Touchstroke-based biometrics"** dataset[cite: 760].
1.  Download the dataset.
2.  Place the dataset file (e.g., `bioident.csv`) into the `data/` directory.

### 6. Load Data into Milvus
Before you can run the VSS server, you must load the BioIdent data into the Milvus database.
```bash
# Make sure the VSS server is NOT running yet.
# Run the one-time data loading script:
python evaluation/load_data.py
```

### 7. 🚨 IMPORTANT: RLWE Crypto Stubs
The post-quantum cryptographic functions in `src/otaka_protocol/helper.py` (e.g., `rlwe_generate_public_key`, `rlwe_compute_shared_values`) are **cryptographic stubs (dummies)**. They mimic the *flow* but provide **no security**.

[cite_start]For a complete implementation, you must replace these stubs with functions from a real lattice-based cryptography library (e.g., one adapted from the repository cited by the paper [cite: 496]).

## How to Run the System

You must run the VSS backend and the OTAKA server in separate terminals.

### 1. Run the VSS Backend (Continuous Authentication)
This server handles the behavioral biometric checks.
```bash
# From the project root directory:
uvicorn src.vss_backend.vss_server:app --reload
```
The API will be live at `http://127.0.0.1:8000`.

### 2. Run the OTAKA Protocol (One-Time Authentication)
This simulates the initial quantum-secure handshake.

**Terminal 1 (Run the Server):**
```bash
python src/otaka_protocol/server.py
```
It will wait for a client to connect.

**Terminal 2 (Run the Client):**
```bash
python src/otaka_protocol/client.py
```
The client will connect, perform the 3-message handshake, and both will terminate, reporting the shared session key.

## How to Run Evaluation & Attack Scripts

These scripts are used to reproduce the paper's results and conduct your security analysis.

### Reproducing Paper Results
(Ensure the VSS backend is running for Fig. 10/11)

```bash
# Reproduce Figure 10 (VSS Accuracy)
python evaluation/reproduce_fig_10_accuracy.py

# Reproduce Figure 11 (VSS Query Speed)
python evaluation/reproduce_fig_11_speed.py

# Reproduce Table II (Crypto Computation Cost)
python evaluation/reproduce_table_2_computation.py
```

### Running Attack Simulations (Project Extension)
(Ensure the respective servers are running)

```bash
# Run Replay Attack on OTAKA server (Section V-B1)
# (Ensure otaka_protocol/server.py is running)
python attacks/attack_replay.py

# Run Impersonation Attack on VSS backend (Section V-B4)
# (Ensure vss_backend/vss_server.py is running)
python attacks/attack_impersonation.py
```