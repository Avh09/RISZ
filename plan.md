# PLAN


### Phase 1: Environment and Data Setup

This initial phase focuses on preparing all the necessary tools, libraries, and data required for the project.

1. **Set Up the Python Environment:**

   * Install Python (version 3.8 or higher is recommended).
   * Create a dedicated virtual environment to manage project dependencies.

2. **Install Required Libraries:**

   * **Cryptography:** Install a standard crypto library for SHA-256 hashing and AES-256 encryption.
   * **Post-Quantum Cryptography (RLWE):** Implement or adapt functions for RLWE operations: polynomial sampling, multiplication, addition, and the `Mod_2`/`Cha` functions.
   * **Vector Database:** Install **Milvus**. This will act as your Feature Vector Database (FVDB). Follow the official Milvus documentation for installation (e.g., using Docker).
   * **API Framework:** Install **FastAPI** and an ASGI server like **Uvicorn**. This is for building the VSS backend.
   * **Data Science & Plotting:** Install `pandas` (for handling the dataset) and `matplotlib` (for reproducing the graphs).

3. **Acquire and Prepare the Dataset:**

   * Download the **BioIdent: Touchstroke-based biometrics on the Android platform** dataset (used for the VSS proof of concept).
   * Familiarize yourself with its structure — it contains records with 15 behavioral features that you will use as your vectors.

---

### Phase 2: Implement the VSS Backend with FastAPI

This phase involves building the continuous authentication service that handles behavioral biometric data. You will replicate the functionality shown in the paper.

1. **Initialize the Vector Database:**

   * Write a script that connects to your Milvus instance.
   * Create a Milvus "collection" (similar to a database table). Configure it to store 15-dimensional vectors and link each vector to a user ID.
   * Create an index on the vector field for fast searching. Use an approximate nearest-neighbor index (e.g., ANNOY) with Euclidean distance.

2. **Create the FastAPI Application:**

   * Set up a new FastAPI Python script.
   * Define the three API endpoints:

     * **`POST /cabb_vector_db_crea`**: Triggers the database initialization script.
     * **`POST /registration_of_behavioral_biometrics_of_a_user`**: Accepts a user ID and a 15-feature vector; inserts the vector and ID into the Milvus collection.
     * **`POST /chk_similarity_of_behavioral_biometrics_of_a_user`**: Accepts a query vector, performs a similarity search in Milvus, and returns the top-matching user ID and similarity score (Euclidean distance).

3. **Populate the Database:**

   * Write a script to read the BioIdent dataset using `pandas`.
   * Iterate through the dataset and use your registration endpoint to load all user vectors into the Milvus database.

---

### Phase 3: Implement the Post-Quantum Authentication Protocol (OTAKA)

This is the core cryptographic part where you build the client-server handshake for the initial login. Follow the protocol logic from the paper.

1. **Set Up Client-Server Scripts:**

   * Create two main scripts: `server_MS.py` and `client_Ui.py`.
   * Use Python's `socket` library for network communication between them.

2. **Implement Server (`server_MS.py`) Logic:**

   * The server should listen for incoming connections.
   * **Step 1: Receive `M1`:** Wait for and parse the first message `M1 = {X1, X2, TID_i, a_i, s_2, TS_1}`.
   * **Step 2: Verify `M1`:** Check timestamp freshness and validate the user identity (for example, compute `ID_i = X1 ⊕ h(t_3 || TS_1 || TID_i)`).
   * **Step 3: Generate and Send `M2`:** If verification succeeds, compute the server’s key-exchange components (e.g., `b_j, c_j, w_j`), calculate the session key `SK_{ji}`, and send `M2 = {SKV_{ji}, TS_2, b_j, d_j, TID_n*}`.
   * **Step 4: Receive and Verify `M3`:** Wait for client acknowledgment `M3 = {ACK, TS_3}` and verify it to finalize the handshake.

3. **Implement Client (`client_Ui.py`) Logic:**

   * The client connects to the server’s address.
   * **Step 1: Generate and Send `M1`:** Compute components for the first message (`a_i, X1, X2, s_2`) and send it to the server.
   * **Step 2: Receive `M2`:** Wait for and parse the server’s reply `M2`.
   * **Step 3: Verify `M2` and Generate Key:** Check the timestamp, compute your version of the session key `SK_{ij}`, and verify server authenticity by ensuring `SKV_{ij} == SKV_{ji}`.
   * **Step 4: Send `M3`:** If the keys match, send the final acknowledgment `M3` to the server.

---

### Phase 4: Integration and Full System Simulation

Connect the OTAKA protocol with the VSS backend to simulate the end-to-end workflow.

1. **Modify Client and Server Scripts:**

   * In `client_Ui.py`, after a successful handshake (session key `SK_{ij}` established), start reading behavioral vectors from the BioIdent dataset.
   * For each vector, encrypt it using AES-256 in CBC mode with the session key `SK_{ij}`, and send the encrypted payload to the server.
   * In `server_MS.py`, after the handshake, listen for encrypted data from the client.

2. **Implement the Continuous Authentication Loop on the Server:**

   * When the server receives encrypted data, decrypt it using the shared session key `SK_{ji}`.
   * The server then forwards the decrypted vector to its FastAPI endpoint via an HTTP `POST` to `/chk_similarity_of_behavioral_biometrics_of_a_user`.
   * If the returned ID matches the authenticated user’s ID, continue the session; otherwise, print "Session Terminated" (simulate detection of an anomaly).

---

### Phase 5: Reproduce Paper's Results

Run experiments to generate the performance and accuracy metrics reported in the paper.

1. **Generate VSS Plots (Figures 10 & 11):**

   * Write an evaluation script that calls your FastAPI endpoints.
   * **For Figure 10:** Query with original vectors and with "shuffled" vectors (created by randomly mixing features between users). Plot distributions of Euclidean distances using `matplotlib`.
   * **For Figure 11:** Run batched query tests to measure search time for blocks of 10, 100, 500, etc., users. Plot the results.

2. **Benchmark Cryptographic Costs (Figure 4):**

   * Create a script that times cryptographic operations (SHA-256, RLWE polynomial multiplication, etc.).
   * Run each operation many times (e.g., 1,000 iterations) and compute average execution times to reproduce the computational cost table.

---

### Phase 6: Extend the Study (Attack Simulations)

Actively test security claims by simulating attacks.

1. **Simulate a Replay Attack:**

   * During a normal authentication, save the raw bytes of the first message `M1`.
   * Create `attacker.py` that sends the saved `M1` to the server.
   * Confirm the server’s timestamp verification rejects the replayed message as stale.

2. **Simulate an Impersonation (Stolen Device) Attack:**

   * Run the integrated simulation and authenticate as "User A".
   * During the continuous authentication loop, instead of sending User A’s vectors, send User B’s vectors.
   * Verify the server’s VSS check detects the mismatch and terminates the session.

---


