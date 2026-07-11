# macOS Setup + Google Cloud (GCP / GKE) Deployment Guide

This guide covers **(1)** running the whole project on macOS, **(2)** creating a
free Google Cloud account, and **(3)** deploying the API to the cloud on GKE
(Google Kubernetes Engine) — plus an easier **Cloud Run** alternative.

> All commands are for the macOS **Terminal** (zsh/bash). Where PENDING.md shows
> PowerShell, use the equivalents below.

**Legend:** 📸 = screenshot for the report · ⚙️ = one-time install · 💸 = uses cloud credit

---

# PART 1 — Run everything on macOS

## 1.1 Install the tools ⚙️

Install [Homebrew](https://brew.sh) first (the macOS package manager):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install everything you need:
```bash
brew install python@3.12 git kubectl minikube helm
brew install --cask docker            # Docker Desktop
brew install --cask google-cloud-sdk  # gcloud CLI (for Part 2/3)
```

Start **Docker Desktop** (open it from Applications once; wait for the whale icon
in the menu bar to go steady). Verify:
```bash
python3 --version      # 3.12.x
docker --version
minikube version
kubectl version --client
helm version
gcloud --version
```

## 1.2 Run the local pipeline

From the project folder:
```bash
cd heart-disease-mlops

python3 -m venv venv
source venv/bin/activate            # (PowerShell was: venv\Scripts\Activate.ps1)
pip install -r requirements.txt

python data/download_data.py        # 📸 "Wrote cleaned dataset"
python -m src.train                 # 📸 3 models + best selected
pytest -v                           # 📸 15 passed
```

## 1.3 MLflow UI 📸
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000
```
Open http://localhost:5000 → capture the runs table, a run's params/metrics, and
its artifacts (ROC + confusion-matrix). `Ctrl+C` to stop.

## 1.4 Serve the API locally 📸
```bash
uvicorn api.main:app --port 8000
```
Open http://localhost:8000/docs → try `/predict`. Test from another terminal:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## 1.5 Docker (local) 📸
```bash
docker build -t heart-api:1.0.0 .
docker run -d --name heart-api -p 8000:8000 heart-api:1.0.0
docker ps
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
docker stop heart-api && docker rm heart-api
```

## 1.6 Monitoring stack (Prometheus + Grafana) 📸
```bash
docker compose up --build -d
# generate some traffic:
for i in $(seq 1 30); do
  curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}' >/dev/null
done
```
Capture: http://localhost:8000/metrics · Prometheus http://localhost:9090
(Status → Targets = UP; run `rate(http_requests_total[1m])`) · Grafana
http://localhost:3000 (`admin`/`admin`) → "Heart Disease API - Monitoring".
Then `docker compose down`.

## 1.7 Local Kubernetes (Minikube) — optional if you do GKE instead 📸
```bash
minikube start --driver=docker
eval $(minikube docker-env)          # (PowerShell was: minikube docker-env | Invoke-Expression)
docker build -t heart-api:1.0.0 .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods && kubectl get svc
minikube service heart-api-service --url    # prints the URL to curl
# cleanup: kubectl delete -f k8s/service.yaml -f k8s/deployment.yaml ; minikube stop
```

## 1.8 GitHub + Actions 📸
Same as PENDING.md §3–4 (git works identically on macOS). In short:
```bash
git init && git add . && git commit -m "Initial commit: heart disease MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/heart-disease-mlops.git
git push -u origin main
```
Then screenshot the green Actions run (and a deliberately failing one).

---

# PART 2 — Create a free Google Cloud account 💸

### The honest facts about "free"
- **Google Cloud Free Trial** gives **$300 in credit for 90 days** to new accounts.
  This easily covers a small GKE cluster for this assignment.
- It **requires a credit/debit card for identity verification**, but you are **NOT
  charged** — your card is only billed if you *manually* upgrade to a paid account.
  A student debit card works.
- There is no simple ".edu = free GCP" program (unlike Azure for Students). Use the
  $300 trial. The **[GitHub Student Developer Pack](https://education.github.com/pack)**
  sometimes bundles extra Google Cloud benefits — worth claiming with your **BITS
  student email**, but the $300 trial is the reliable path.

> 🔁 **Easier no-card alternative:** if you have a `@*.edu`/`@*.ac.in` student email,
> **Azure for Students** gives **$100 with NO credit card**. Your k8s manifests would
> deploy to **AKS** with the same `kubectl apply`. Mentioned here only as a fallback —
> the rest of this guide uses GCP.

### Steps to create the account
1. Go to **https://cloud.google.com/free** → click **Get started for free**.
2. Sign in with a Google account (use your student Gmail if you have one).
3. Choose **Country = India**, accept terms.
4. Account type: **Individual**.
5. Add a **payment method** (card). You'll see a small temporary verification hold
   that is refunded. 📸 (optional) the "$300 credit" confirmation screen.
6. You land in the **Google Cloud Console** (https://console.cloud.google.com).
7. Create a project: top bar → **project dropdown → New Project** → name it
   `heart-mlops` → **Create**. Note the **Project ID** (e.g. `heart-mlops-471203`).

### ⚠️ Avoid surprise charges
- After you finish and take screenshots, **delete the GKE cluster** (Part 3.7) so it
  stops consuming credit.
- Set a **budget alert**: Console → Billing → Budgets & alerts → create a ₹500 / $5
  alert so you're emailed if anything runs.

---

# PART 3 — Deploy to Google Cloud

You have two options. **Option A (GKE)** matches the assignment best (uses your
Kubernetes manifests + a real LoadBalancer). **Option B (Cloud Run)** is the
fastest way to a public URL. Do **A** for full marks; B is a backup.

## First: authenticate the gcloud CLI (both options)
```bash
gcloud auth login                                  # opens a browser
gcloud config set project <PROJECT_ID>             # from Part 2 step 7
gcloud auth configure-docker REGION-docker.pkg.dev # e.g. us-central1-docker.pkg.dev
```
Pick a region close to you, e.g. `asia-south1` (Mumbai) or `us-central1`. Use the
**same region** everywhere below.

Enable the required APIs (one-time):
```bash
gcloud services enable container.googleapis.com artifactregistry.googleapis.com run.googleapis.com
```

---

## OPTION A — GKE (Google Kubernetes Engine) 💸 📸

### A.1 Create an Artifact Registry repo and push the image
```bash
export REGION=asia-south1           # or us-central1
export PROJECT_ID=<PROJECT_ID>

gcloud artifacts repositories create heart-repo \
  --repository-format=docker --location=$REGION \
  --description="Heart API images"

# Build for linux/amd64 (important on Apple-Silicon Macs, which are arm64):
docker build --platform linux/amd64 -t heart-api:1.0.0 .

# Tag + push to Artifact Registry:
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/heart-repo/heart-api:1.0.0
docker tag heart-api:1.0.0 $IMAGE
docker push $IMAGE                  # 📸 successful push
echo $IMAGE                         # copy this full path
```

### A.2 Create a GKE cluster
Autopilot is simplest (Google manages nodes; billed per pod resource):
```bash
gcloud container clusters create-auto heart-cluster --region $REGION   # 📸 (takes ~5-8 min)
gcloud container clusters get-credentials heart-cluster --region $REGION
kubectl config current-context      # should now point at the GKE cluster
```
*(Cheaper fixed alternative — a tiny Standard cluster:
`gcloud container clusters create heart-cluster --region $REGION --num-nodes=1 --machine-type=e2-small`)*

### A.3 Point the manifest at your image and deploy
Edit `k8s/gke/deployment.yaml` and replace the `image:` line with the `$IMAGE`
value from A.1 (or do it inline with sed):
```bash
sed -i '' "s#REGION-docker.pkg.dev/PROJECT_ID/heart-repo/heart-api:1.0.0#$IMAGE#" k8s/gke/deployment.yaml

kubectl apply -f k8s/gke/deployment.yaml
kubectl apply -f k8s/gke/service.yaml

kubectl get pods -w                 # 📸 wait until both are Running (Ctrl+C to stop watching)
```

### A.4 Get the public URL and test 📸
```bash
kubectl get service heart-api-service     # wait for EXTERNAL-IP to appear (1-2 min)
export EXTIP=$(kubectl get svc heart-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "http://$EXTIP"

curl http://$EXTIP/health                 # 📸
curl -X POST http://$EXTIP/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'   # 📸
```
Also open `http://$EXTIP/docs` in a browser → 📸 Swagger UI on the **public cloud**.
This public URL is your **"Deployed API URL"** deliverable.

### A.5 Extra screenshots for the report 📸
```bash
kubectl get all                     # 📸 all resources
kubectl describe deployment heart-api
```
Also in the **Cloud Console**: Kubernetes Engine → Workloads (📸) and
Services & Ingress (📸 showing the external endpoint).

### A.6 (Optional) Helm on GKE
```bash
helm install heart k8s/helm/heart-api \
  --set image.repository=$REGION-docker.pkg.dev/$PROJECT_ID/heart-repo/heart-api \
  --set image.tag=1.0.0 --set image.pullPolicy=Always
kubectl get all         # 📸
# helm uninstall heart
```

### A.7 🧹 CLEAN UP (do this when done to stop spending credit!)
```bash
kubectl delete -f k8s/gke/service.yaml -f k8s/gke/deployment.yaml
gcloud container clusters delete heart-cluster --region $REGION
# optional: gcloud artifacts repositories delete heart-repo --location=$REGION
```

---

## OPTION B — Cloud Run (fastest public URL, no Kubernetes) 💸 📸

Great as a backup or to *also* have a live serverless URL. One command builds,
pushes and deploys:
```bash
gcloud run deploy heart-api \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --port 8000
```
It prints a public HTTPS URL, e.g. `https://heart-api-xxxx.a.run.app`. Test:
```bash
curl -X POST https://heart-api-xxxx.a.run.app/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```
📸 the URL + response, and the Cloud Run service page in the Console.
Cleanup: `gcloud run services delete heart-api --region $REGION`.

> ⚠️ Cloud Run does **not** use your Kubernetes manifests / LoadBalancer, so it only
> partially matches Task 7's wording. If you can, do **Option A (GKE)** for the K8s
> requirement and keep Cloud Run as a bonus.

---

# Which path should I pick?

| Goal | Recommendation |
|------|----------------|
| Full marks on Task 7 (K8s + manifests + LoadBalancer) | **GKE (Option A)** |
| Just need a live public URL fast | **Cloud Run (Option B)** |
| No credit card available | **Minikube locally** (Part 1.7) — still fully valid per the FAQ |
| Have a student `.edu`/`.ac.in` email, want no card | **Azure for Students → AKS** (same manifests) |

For the report, whichever you choose: capture the deploy commands, the running
pods/service, the **public URL**, and a successful `/predict` response, then paste
them into the labelled placeholders in `report/MLOps_Assignment01_Report.docx`.
