# redhat-event


# Implementation Guide: UPI Anomaly Detection on OpenShift

1. **Set up the OpenShift environment:** After receiving cluster credentials, log into your OpenShift cluster. For example, use the OpenShift CLI:

   ```bash
   oc login --token=<your-token> --server=<cluster-URL>
   ```

   You can also access the OpenShift Web Console via the provided URL and token. Once logged in, verify your access by running `oc whoami` or `oc get nodes`. This ensures you can create projects and deploy apps.

2. **Create a project (namespace):** In Kubernetes/OpenShift, a *namespace* scopes resources (pods, services, etc.) so they don’t collide with others. In OpenShift, a *project* is basically a Kubernetes namespace with extra metadata for user access control. Create one for your hackathon app, e.g.:

   ```bash
   oc new-project upi-anomaly-demo
   ```

   This isolates your team’s resources. You can list projects with `oc get projects`. Think of the project as your sandbox: all deployments, services, and other objects live here. Inside it, you have unique names (pods, services) that won’t conflict with other projects, and you’re the admin of this project by default.

3. **Generate synthetic UPI transaction data:** Write a simple Python script to simulate UPI transactions. Include fields like user ID, amount, timestamp, location, etc. For example:

   ```python
   import random, datetime, uuid

   def random_transaction():
       return {
           "txn_id": str(uuid.uuid4()),
           "user_id": random.randint(1, 100),
           "amount": round(random.uniform(10.0, 5000.0), 2),
           "timestamp": datetime.datetime.now().isoformat(),
           "merchant": random.choice(["Grocery", "Travel", "Utilities", "Retail"]),
           "location": random.choice(["Delhi", "Mumbai", "Bengaluru", "Kolkata"])
       }

   # Generate a list of transactions (including some large ones to simulate anomalies)
   data = [random_transaction() for _ in range(1000)]
   for tx in data[:5]:
       print(tx)
   ```

   This creates normal and some unusual transactions (e.g. very large `amount`) that can be treated as anomalies. Save this data to use for model “training” or testing. (Optionally, you can use public synthetic datasets or more advanced generators, but random sampling is simplest for a hackathon.)

4. **Load and use pre-trained models:** We will use two anomaly detectors: an Isolation Forest and an Autoencoder. These can be created with libraries like PyOD (which wraps scikit-learn and PyTorch models). For example, an Isolation Forest isolates anomalies with fewer random splits, while an Autoencoder is a neural network that flags outliers by high reconstruction error. In Python, you might do:

   ```python
   from pyod.models.iforest import IForest
   from pyod.models.auto_encoder import AutoEncoder

   # Assume X_train is your training data (e.g. transaction features)
   if_model = IForest(contamination=0.01)
   if_model.fit(X_train)

   ae_model = AutoEncoder(hidden_neurons=[64, 32], epochs=50, contamination=0.01)
   ae_model.fit(X_train)
   ```

   After fitting (offline on your synthetic data), save the models (e.g. using `joblib.dump(if_model, "iforest.pkl")`). In your scoring service, you will load these models and call `model.predict()` on new transactions. (Because IsolationForest and AutoEncoder assign higher anomaly scores to outliers, you can use their `predict` or `decision_function` results.) This lets you flag whether each transaction is normal or anomalous. You don’t need to train complex models in the hackathon, just load these pre-fit models for inference.

5. **Develop and containerize microservices:** We will build small Python services (using frameworks like FastAPI or Flask) for different parts of the pipeline. *FastAPI* is ideal for high-speed REST APIs, and *Flask* is easy to use for simple services. For example:

   &#x20;*Create a simple FastAPI service for ingestion:*

   ```python
   # ingest_service.py
   from fastapi import FastAPI
   import requests

   app = FastAPI()
   SCORING_URL = "http://scoring-service/score"

   @app.post("/transaction")
   def ingest(tx: dict):
       # Forward transaction to the scoring service
       requests.post(SCORING_URL, json=tx)
       return {"status": "transaction received"}
   ```

   Build this into a Docker image (`FROM python:3-slim`, install FastAPI+uvicorn) and expose port 8080.

   * **Scoring service:** A separate FastAPI (or Flask) app loads the saved models and checks each transaction. For example:

     ```python
     # scoring_service.py
     from fastapi import FastAPI
     import joblib

     app = FastAPI()
     if_model = joblib.load("iforest.pkl")
     ae_model = joblib.load("autoencoder.pkl")

     @app.post("/score")
     def score(tx: dict):
         # Extract features (e.g. [amount, ...]) from tx
         features = [tx["amount"], ...] 
         # Get anomaly labels (1 = outlier in PyOD)
         out_if = if_model.predict([features])[0]
         out_ae = ae_model.predict([features])[0]
         anomaly = (out_if == 1) or (out_ae == 1)
         if anomaly:
             print("Anomaly detected:", tx)
         return {"user_id": tx["user_id"], "anomaly": bool(anomaly)}
     ```

     This logs anomalies to the service console. You can also connect to Redis or a database if desired, but printing to console (captured by OpenShift logs) is simplest.

   * **Explainability (optional):** If you want to add SHAP, create a small helper. SHAP can explain predictions by giving feature contributions. For example:

     ```python
     import shap
     # After scoring
     explainer = shap.TreeExplainer(if_model)  # works for tree-based models
     shap_values = explainer.shap_values([features])
     print("Feature contributions:", shap_values)
     ```

     This is optional; you can simply log the SHAP values or include them in the response for one transaction to show why it was flagged.

   * **Alert service:** Finally, you can have a simple service that sends alerts. For example:

     ```python
     # alert_service.py
     from fastapi import FastAPI
     app = FastAPI()

     @app.post("/alert")
     def alert(tx: dict):
         print(f"*** ALERT: Fraudulent transaction flagged: {tx}")
         return {"notified": True}
     ```

     In practice, your scoring service could call this alert endpoint when an anomaly is found. (For a hackathon, even printing to its own log is fine.)

   Containerize each service with a Dockerfile. For example, a FastAPI Dockerfile might be:

   ```Dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD ["uvicorn", "ingest_service:app", "--host", "0.0.0.0", "--port", "8080"]
   ```

   Build and tag (`docker build -t quay.io/<org>/ingest:latest .`) and similarly for scoring/alert services. This produces container images you can push to a registry.

6. **Deploy containers on OpenShift:** First, push your Docker images to a registry that OpenShift can pull from (e.g. Red Hat Quay, Docker Hub, or the integrated OpenShift registry). Red Hat Quay is a secure, scalable private registry (often integrated into OpenShift). For example:

   ```bash
   docker push quay.io/<org>/ingest:latest
   docker push quay.io/<org>/scoring:latest
   docker push quay.io/<org>/alert:latest
   ```

   In OpenShift, deploy each image. You can use `oc new-app` or apply Kubernetes YAML. For instance:

   ```bash
   oc new-app --docker-image=quay.io/<org>/ingest:latest --name=ingest-service
   oc expose svc/ingest-service --port=8080
   ```

   Repeat for the scoring and alert services. This creates Deployments, Pods, and Services. You might also need a Service for internal communication (ingest → scoring, scoring → alert). Alternatively, you can define Deployment and Service YAML files and run `oc apply -f`. For example, a Deployment manifest for FastAPI might include the `image: quay.io/<org>/scoring:latest` and a Service that routes port 80 to container port 8080. OpenShift will pull your image and start the pods.

   (As an advanced option, you could set up an **OpenShift Pipeline** (Tekton) to build and deploy automatically on git push, but for a quick POC it’s fine to do manual builds/deploys. The key is that OpenShift’s `oc new-app` and deployment configs will orchestrate the containers for you.)

7. **Test the system end-to-end:** Once deployed, test by sending sample transactions. For example, use `curl` or a simple Python script to POST to the ingestion endpoint:

   ```bash
   curl -X POST http://<ingest-route>/transaction -H "Content-Type: application/json" \
        -d '{"user_id": 42, "amount": 1234.56, "merchant": "Travel", "location": "Delhi"}'
   ```

   Check that the scoring service receives it (its logs should show the transaction and any anomaly messages). If an anomaly is detected, you should see the alert service log output (or whichever method you used). You can also scale the number of pods to simulate load, and use `oc get pods` / `oc logs <pod>` to inspect behavior. This verifies that data flows from ingestion → scoring → alert as expected.

8. **Create a simple dashboard or HTML page:** For a quick dashboard, you can serve a static HTML or a small web app. For example, spin up another Python service (Flask or FastAPI with templates) that reads flagged transactions (e.g. from a shared JSON or database) and displays them. A very simple approach: have the scoring service append anomalies to a file or push them to a shared Redis; then a dashboard service reads that and renders an HTML table. Pseudo-code:

   ```python
   # dashboard_service.py
   from flask import Flask, render_template
   import json

   app = Flask(__name__)
   @app.route("/")
   def show_dashboard():
       with open("anomalies.json") as f:
           anomalies = json.load(f)
       return render_template("dashboard.html", anomalies=anomalies)
   ```

   In `dashboard.html`, loop over `anomalies` to show fields like user, amount, time. The goal is a quick visual: e.g. a page listing “Anomalous Transactions” or a count graph. (You could also use a notebook or simple Grafana chart if available, but a basic HTML page meets the requirement.) This lets judges see your flagged transactions in a user-friendly way.

9. **Capture the demo and document:**  Walk through: setting up the project, showing the code for each service, deploying the pods in OpenShift, and then triggering a transaction to see the alert and dashboard update. Narrate the flow: ingestion → scoring (with models) → alert and dashboard. Also prepare brief documentation summarizing your steps (for example, a README or slides). Include pointers to how to re-run the code and any caveats. The video can simply be a screen recording (Zoom, OBS, etc.) where you explain each component. Remember that OpenShift access is only for 4 weeks, so document everything clearly.

Each of these steps uses only basic, pre-packaged tools (FastAPI/Flask, PyOD models, OpenShift CLI) so you don’t have to build ML models from scratch. By following this guide, you’ll have a working POC of real-time UPI fraud detection: synthetic data flowing into your microservices, anomalies flagged by Isolation Forest and AutoEncoder, and results displayed on a dashboard. Good luck and happy hacking!

**Sources:** We explained Isolation Forest and AutoEncoder concepts based on standard references, and SHAP’s role in explaining predictions. Our steps for Python microservices and OpenShift deployment follow best practices. We also noted that OpenShift projects are Kubernetes namespaces with access control and that Red Hat Quay is a private container registry integrated with OpenShift.
