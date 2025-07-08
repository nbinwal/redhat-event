**UPI Fraud Shield AI POC on Red Hat OpenShift**

A hands‑on, step‑by‑step guide to build and deploy a UPI Fraud Detection POC using the project `pes-a-upi-fraud-detector-2025`. Designed for newcomers to OpenShift and AI, this document will "spoon‑feed" every command, YAML snippet, and configuration you need.

---

## 1. Prerequisites

1. **OpenShift Access**: Ensure you have cluster-admin or project-admin rights.
2. **CLI Tools**:

   * Install & login:

     ```bash
     oc version
     oc login --token=<YOUR_TOKEN> --server=<API_URL>
     ```
   * Optional: `kubectl`, `tkn` (Tekton CLI).
3. **Git Repo Setup**:

   * Create a GitHub repo (e.g. `upi-fraud-demo`).
   * Clone locally:

     ```bash
     git clone https://github.com/your-org/upi-fraud-demo.git
     cd upi-fraud-demo
     ```
4. **Dummy Data**:

   * Ensure `dummy_upi_transactions.csv` is in `data/` folder locally.
5. **Project Namespace**: We’ll use **`pes-a-upi-fraud-detector-2025`**.

   ```bash
   oc new-project pes-a-upi-fraud-detector-2025 || oc project pes-a-upi-fraud-detector-2025
   ```

---

## 2. Install Required Operators

### 2.1. OpenShift Data Science (AI)

* Console → **Operators → OperatorHub** → Search **"OpenShift Data Science"** → Install in `openshift-operators`.

### 2.2. OpenShift Pipelines (Tekton)

* OperatorHub → Search **"OpenShift Pipelines"** → Install in `openshift-pipelines`.

### 2.3. Streams for Apache Kafka

* OperatorHub → Search **"Streams for Apache Kafka"** → Install in `openshift-operators`.

### 2.4. User Workload Monitoring

* Console → **Administrator → Cluster Settings** → Edit **Cluster** → Enable **User Workload Monitoring** → Save.

---

## 3. Prepare Storage & Data

### 3.1. Create PVC

Save as `pvc-upi-data.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: upi-data-pvc
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 1Gi
```

```bash
oc apply -f pvc-upi-data.yaml
```

### 3.2. Upload CSV into PVC

```bash
# 1. Launch loader pod with PVC mounted
oc run loader \
  --image=registry.access.redhat.com/ubi8/ubi \
  --restart=Never \
  --overrides='
{
  "apiVersion":"v1","kind":"Pod",
  "spec":{
    "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"upi-data-pvc"}}],
    "containers":[
      {
        "name":"loader",
        "image":"registry.access.redhat.com/ubi8/ubi",
        "command":["sleep","3600"],
        "volumeMounts":[{"mountPath":"/data","name":"data"}]
      }
    ],
    "restartPolicy":"Never"
  }
}
'
# 2. Copy file from local into pod
tar cf - -C data dummy_upi_transactions.csv | oc exec -i pod/loader -- tar xf - -C /data
# 3. Inside pod verify and exit
tail /data/dummy_upi_transactions.csv
oc delete pod loader
```

Now your CSV lives at `/data/dummy_upi_transactions.csv` on the PVC.

---

## 4. Model Training in OpenShift Data Science

### 4.1. Launch JupyterLab

1. Console → **OpenShift AI → Data Science Projects** → Select **pes-a-upi-fraud-detector-2025**.
2. Click **Launch standalone workbench**.
3. In JupyterLab, attach the PVC to a new notebook server: Notebook → Attach Storage → `upi-data-pvc` → mount at `/data`.

### 4.2. Create & Run Notebook

In JupyterLab, New → Python 3 → rename to `train_fraud_model.ipynb`. Copy-paste and run cells:

```python
# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib, shap

# 2. Load data
df = pd.read_csv('/data/dummy_upi_transactions.csv')

# 3. Feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['sender_dom'] = df['sender_vpa'].str.split('@').str[1]
df['recv_dom']   = df['receiver_vpa'].str.split('@').str[1]
X = pd.get_dummies(df[['amount','location','device_id','hour','sender_dom','recv_dom']])
y = df['is_fraud']

# 4. Train/Test split & model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test,y_test))

# 5. SHAP explainer
explainer = shap.TreeExplainer(model)

# 6. Save model & explainer
!mkdir -p /data/model
joblib.dump(model, '/data/model/fraud_model.joblib')
joblib.dump(explainer, '/data/model/shap_explainer.joblib')
```

Notebook will train and save artifacts to `/data/model/` on the PVC.

---

## 5. FastAPI Service: Code, Build, Deploy

### 5.1. Service Code

In your cloned Git repo under `fraud-api/`, create three files:

**`fraud-api/main.py`**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd
import shap
from prometheus_client import Counter, start_http_server

app = FastAPI()
# Load artifacts
model = joblib.load('/opt/model/fraud_model.joblib')
explainer = joblib.load('/opt/model/shap_explainer.joblib')
features = model.feature_names_in_
# Prometheus metric
fraud_counter = Counter('fraud_detected_total','Total fraud alerts')
start_http_server(8000)

class Tx(BaseModel):
    amount: float
    location: str
    device_id: str
    timestamp: str
    sender_vpa: str
    receiver_vpa: str

@app.post('/predict')
def predict(tx: Tx):
    df = pd.DataFrame([tx.dict()])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['sender_dom'] = df['sender_vpa'].split('@')[1]
    df['recv_dom']   = df['receiver_vpa'].split('@')[1]
    X = pd.get_dummies(df[['amount','location','device_id','hour','sender_dom','recv_dom']])
    X = X.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(X)[0,1]
    pred = bool(model.predict(X)[0])
    if pred:
        fraud_counter.inc()
    # SHAP local values
    shap_vals = explainer.shap_values(X)[1][0]
    contrib = dict(sorted(zip(features, shap_vals), key=lambda kv: abs(kv[1]), reverse=True)[:3])
    return {'is_fraud': pred, 'probability': prob, 'explanation': contrib}
```

**`fraud-api/requirements.txt`**

```
fastapi
uvicorn
scikit-learn
joblib
pandas
shap
prometheus_client
```

**`fraud-api/Dockerfile`**

```dockerfile
FROM registry.access.redhat.com/ubi8/python-39
WORKDIR /opt/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]
```

Commit & push:

```bash
git add fraud-api/
git commit -m "Add fraud API service"
git push
```

### 5.2. Build & ImageStream

Save as `build-fraud-api.yaml`:

```yaml
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  name: fraud-api
---
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: fraud-api
spec:
  source:
    type: Git
    git:
      uri: 'https://github.com/your-org/upi-fraud-demo.git'
      contextDir: 'fraud-api'
  strategy:
    type: Docker
  output:
    to:
      kind: ImageStreamTag
      name: 'fraud-api:latest'
```

```bash
oc apply -f build-fraud-api.yaml
oc start-build fraud-api --follow
```

### 5.3. Deployment, Service & Route

Save as `deploy-fraud-api.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-api
spec:
  replicas: 2
  selector:
    matchLabels: { app: fraud-api }
  template:
    metadata:
      labels: { app: fraud-api }
    spec:
      containers:
      - name: fraud-api
        image: image-registry.openshift-image-registry.svc:5000/pes-a-upi-fraud-detector-2025/fraud-api:latest
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: model-vol
          mountPath: /opt/model
      volumes:
      - name: model-vol
        persistentVolumeClaim:
          claimName: upi-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-api-svc
spec:
  selector:
    app: fraud-api
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: fraud-api-route
spec:
  to:
    kind: Service
    name: fraud-api-svc
  port:
    targetPort: 8080
```

```bash
oc apply -f deploy-fraud-api.yaml
```

Now your API is live at `http://fraud-api-route-pes-a-upi-fraud-detector-2025.apps.<cluster>/predict`.

---

## 6. Simulate Real-Time Scoring via Kafka

### 6.1. Deploy Kafka Cluster

Save `kafka-cluster.yaml`:

```yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: upi-kafka
spec:
  kafka:
    replicas: 1
    listeners:
    - name: plain
      port: 9092
      type: internal
    storage:
      type: ephemeral
  zookeeper:
    replicas: 1
    storage:
      type: ephemeral
```

```bash
oc apply -f kafka-cluster.yaml
```

### 6.2. Create Topic

Save `kafka-topic.yaml`:

```yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: upi-transactions
  labels:
    strimzi.io/cluster: upi-kafka
spec:
  partitions: 3
  replicas: 1
```

```bash
oc apply -f kafka-topic.yaml
```

### 6.3. Producer App & CronJob

#### 6.3.1. Producer Code

In repo `producer/producer.py`:

```python
import json, time
from kafka import KafkaProducer
import pandas as pd

def main():
    df = pd.read_csv('/data/dummy_upi_transactions.csv')
    producer = KafkaProducer(
        bootstrap_servers='upi-kafka-bootstrap:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    for _, row in df.iterrows():
        producer.send('upi-transactions', row.to_dict())
        time.sleep(1)
    producer.flush()

if __name__=='__main__':
    main()
```

Create `producer/Dockerfile`:

```dockerfile
FROM registry.access.redhat.com/ubi8/python-39
RUN pip install kafka-python pandas
WORKDIR /opt/producer
COPY producer.py .
CMD ["python","producer.py"]
```

Commit & push, then build an ImageStream & BuildConfig similarly to section 5.2 (call it `upi-producer`).

#### 6.3.2. CronJob

Save `producer-cronjob.yaml`:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: produce-upi
spec:
  schedule: "*/1 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: producer
            image: image-registry.openshift-image-registry.svc:5000/pes-a-upi-fraud-detector-2025/upi-producer:latest
            volumeMounts:
            - name: data
              mountPath: /data
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: upi-data-pvc
          restartPolicy: OnFailure
```

```bash
oc apply -f producer-cronjob.yaml
```

### 6.4. Scoring Consumer App

Similarly, create a `consumer/` folder:

**`consumer/consumer.py`**

```python
import json
from kafka import KafkaConsumer, KafkaProducer
import requests

def main():
    consumer = KafkaConsumer(
        'upi-transactions',
        bootstrap_servers='upi-kafka-bootstrap:9092',
        value_deserializer=lambda m: json.loads(m)
    )
    producer = KafkaProducer(
        bootstrap_servers='upi-kafka-bootstrap:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    for msg in consumer:
        txn = msg.value
        resp = requests.post(
            'http://fraud-api-svc:8080/predict', json=txn
        )
        result = resp.json()
        # push result to new topic
        producer.send('upi-results', result)
        # optional: print or log
        print(result)

if __name__=='__main__':
    main()
```

**`consumer/Dockerfile`**

```dockerfile
FROM registry.access.redhat.com/ubi8/python-39
RUN pip install kafka-python requests
WORKDIR /opt/consumer
COPY consumer.py .
CMD ["python","consumer.py"]
```

Build & deploy as Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: upi-consumer
spec:
  replicas: 1
  selector:
    matchLabels: { app: upi-consumer }
  template:
    metadata:
      labels: { app: upi-consumer }
    spec:
      containers:
      - name: consumer
        image: image-registry.openshift-image-registry.svc:5000/pes-a-upi-fraud-detector-2025/upi-consumer:latest
```

```bash
oc apply -f consumer-deploy.yaml
```

Now end‑to‑end: Kafka → producer → API → consumer → results.

---

## 7. Monitoring & Alerts

### 7.1. Expose Metrics (already in section 5)

Metrics on port 8000.

### 7.2. ServiceMonitor

Save `servicemonitor.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fraud-api-monitor
  labels: { team: fraud }
spec:
  selector: { matchLabels: { app: fraud-api } }
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
```

```bash
oc apply -f servicemonitor.yaml
```

### 7.3. Grafana Dashboard

1. Console → Observability → Grafana → Import → paste JSON. Use metrics:

   * `sum(fraud_detected_total)`
   * `rate(http_requests_total[1m])`
   * CPU/Memory via built‑in dashboards.

---

## 8. CI/CD with Tekton Pipelines

### 8.1. Task Definitions

Save `tekton-tasks.yaml`:

```yaml
apiVersion: tekton.dev/v1beta1
kind: Task
metadata:
  name: train-model
spec:
  workspaces:
  - name: shared-data
  steps:
  - name: train
    image: quay.io/centos7/python-36
    script: |
      pip install pandas scikit-learn joblib shap
      python /workspace/shared-data/script/train.py

---
apiVersion: tekton.dev/v1beta1
kind: Task
metadata:
  name: build-image
spec:
  resources:
    inputs:
    - name: source
      type: git
  steps:
  - name: build
    image: gcr.io/kaniko-project/executor:latest
    args:
    - --dockerfile=/workspace/source/fraud-api/Dockerfile
    - --context=/workspace/source/fraud-api
    - --destination=image-registry.openshift-image-registry.svc:5000/pes-a-upi-fraud-detector-2025/fraud-api:$(params.TAG)
```

### 8.2. Pipeline Definition

Save `upi-pipeline.yaml`:

```yaml
apiVersion: tekton.dev/v1beta1
kind: Pipeline
metadata:
  name: upi-train-pipeline
spec:
  params:
  - name: TAG
    default: 'latest'
  resources:
  - name: repo
    type: git
  workspaces:
  - name: shared-data
  tasks:
  - name: train
    taskRef:
      name: train-model
    workspaces:
    - name: shared-data
      workspace: shared-data
  - name: build
    taskRef:
      name: build-image
    resources:
      inputs:
      - name: source
        resource: repo
    params:
    - name: TAG
      value: $(params.TAG)
```

```bash
oc apply -f tekton-tasks.yaml
oc apply -f upi-pipeline.yaml
```

### 8.3. Trigger Setup

Save `tekton-trigger.yaml`:

```yaml
apiVersion: triggers.tekton.dev/v1alpha1
kind: TriggerBinding
metadata:
  name: on-git
spec:
  params:
  - name: gitrevision
    value: $(body.head_commit.id)
---
apiVersion: triggers.tekton.dev/v1alpha1
kind: TriggerTemplate
metadata:
  name: pipeline-template
spec:
  params:
  - name: gitrevision
  resourcetemplates:
  - apiVersion: tekton.dev/v1beta1
    kind: PipelineRun
    metadata:
      generateName: upi-run-
    spec:
      pipelineRef:
        name: upi-train-pipeline
      params:
      - name: TAG
        value: $(params.gitrevision)
      resources:
      - name: repo
        resourceRef:
          name: upi-demo-repo
---
apiVersion: triggers.tekton.dev/v1alpha1
kind: EventListener
metadata:
  name: git-listener
spec:
  triggers:
  - name: git-trigger
    bindings:
    - ref: on-git
    template:
      ref: pipeline-template
```

```bash
oc apply -f tekton-trigger.yaml
```

Now pushing to Git will start the pipeline.

---

## 9. Drift Detection & Automated Retraining

### 9.1. Drift Job Script

Add `drift_check.py` in repo:

```python
# sample code: load recent predictions, compare to dummy labels, compute accuracy
```

### 9.2. CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: drift-monitor
spec:
  schedule: "0 2 * * *" # daily at 2AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: drift
            image: quay.io/your-org/drift-checker:latest
            args: ['python','drift_check.py']
            env:
            - name: PIPELINE_URL
              value: "http://tekton-pipelines.tekton.svc/pipelineruns"
          restartPolicy: OnFailure
```

Inside `drift_check.py`, if accuracy<0.8, call:

```bash
tkn pipeline start upi-train-pipeline --param TAG=auto
```

---

## 10. Validation & Testing

1. **API Smoke Test**:

   ```bash
   curl -X POST http://fraud-api-route-pes-a-upi-fraud-detector-2025.apps.<cluster>/predict \
     -H 'Content-Type: application/json' \
     -d '{"amount":5000,"location":"X","device_id":"dev1","timestamp":"2025-07-01T12:00:00","sender_vpa":"a@b","receiver_vpa":"c@d"}'
   ```
2. **Metrics Check**:

   ```bash
   oc port-forward svc/fraud-api-metrics 8000:8000 &
   curl localhost:8000/metrics | grep fraud_detected_total
   ```
3. **Kafka Flow**: Verify `upi-results` topic messages:

   ```bash
   oc exec -it deployment/upi-consumer -- kafka-console-consumer.sh --bootstrap-server upi-kafka-bootstrap:9092 --topic upi-results --from-beginning
   ```
4. **Pipeline Run**:

   ```bash
   tkn pipelinerun list
   ```


5. **Grafana Dashboard**: Confirm graphs for fraud alerts and API latency.

The POC is now end‑to‑end: data ingestion on PVC/Kafka, model training on Data Science, CI/CD with Tekton, real‑time scoring via FastAPI & Kafka, explainability with SHAP, monitoring + dashboards, and automated retraining.
