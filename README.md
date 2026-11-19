# Typology — MBTI Personality Inference Web App

An editorial, zero-emoji web application for four-axis MBTI personality prediction from written language (`I/E`, `N/S`, `T/F`, `J/P`).

## Project Structure

- `app.py` — FastAPI server and REST endpoints (`/api/predict`, `/api/status`, `/api/types`, `/api/upload-model`).
- `model_engine.py` — Text cleaning, 16 length-invariant stylometric features, ML ensemble loader, and built-in psycholinguistic fallback estimator.
- `train.py` — Standalone CLI script to train `mbti_predictor.joblib` locally from `mbti_1.csv.zip` or `mbti_1.csv`.
- `MBTI_prediction_improved.ipynb` — Improved Colab/Jupyter notebook that trains and exports the portable `mbti_predictor.joblib` bundle.
- `static/index.html`, `static/styles.css`, `static/app.js` — Editorial single-page frontend.
- `run_webapp.bat` — One-click launcher for Windows.

## Quick Start

1. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python app.py
   ```
   Or double-click `run_webapp.bat`.
3. Open `http://127.0.0.1:8000` in your browser.

## Connecting Your Trained ML Ensemble

The app works immediately using a built-in psycholinguistic estimator so you can test the interface right away. To upgrade to the full 8,675-user TF-IDF + Stylometry + LightGBM ensemble:

1. Run `MBTI_prediction_improved.ipynb` (through Cell 16) in Google Colab and download `mbti_predictor.joblib`, **or** run locally:
   ```bash
   python train.py --data mbti_1.csv.zip
   ```
2. Either place `mbti_predictor.joblib` in this folder and restart `app.py`, **or** click the **Estimator Mode** button in the top-right header of the web app to upload `mbti_predictor.joblib` with instant hot-reload.
