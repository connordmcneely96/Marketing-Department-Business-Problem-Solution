# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Your Data
Place your dataset CSV in `data/raw/data.csv`

### 3. Train Your Model
```bash
python -m src.train
```

---

## 📋 Common Tasks

### Run Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```

### Run FastAPI Server
```bash
cd api
uvicorn main:app --reload
```

### Make a Prediction (Python)
```python
from src.inference import predict_classification

features = {
    "feature_1": 1.0,
    "feature_2": 2.0,
    # ... add your features
}

result = predict_classification(features, return_proba=True)
print(result)
```

### Make a Prediction (API)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 1.0, "feature_2": 2.0}}'
```

---

## 🔧 Customization Checklist

When starting a new project:

### 1. Update `src/config.py`
- [ ] Set `RAW_DATA_FILE` name
- [ ] Define `NUMERIC_FEATURES` list
- [ ] Define `CATEGORICAL_FEATURES` list
- [ ] Set `TARGET_COLUMN` name
- [ ] Adjust `TEST_SIZE` and `RANDOM_SEED`

### 2. Adapt Preprocessing (`src/preprocess.py`)
- [ ] Customize `handle_missing_values()` strategy
- [ ] Update `encode_categorical_features()` for your data
- [ ] Modify `scale_numeric_features()` if needed

### 3. Configure Training (`src/train.py`)
- [ ] Choose models in `get_classification_models()`
- [ ] Update hyperparameters in `src/config.py`
- [ ] Set `stratify=True` if dealing with imbalanced classes
- [ ] Change `model_type` parameter if not classification

### 4. Customize Streamlit App (`app/streamlit_app.py`)
- [ ] Update page title and description
- [ ] Change input controls to match your features
- [ ] Customize prediction display format
- [ ] Add business-friendly explanations

### 5. Update README.md
- [ ] Fill in project title and business problem
- [ ] Describe dataset
- [ ] Add model results
- [ ] Document business impact
- [ ] Add demo URLs after deployment

---

## 🎯 Project Types Quick Reference

### Classification
- Models: LogisticRegression, RandomForest, XGBoost
- Metrics: accuracy, precision, recall, f1, roc_auc
- Demo: Feature inputs → Predicted class + probabilities

### Regression
- Models: LinearRegression, RandomForest, XGBoost
- Metrics: mae, mse, rmse, r2, mape
- Demo: Feature inputs → Predicted value

### Clustering
- Models: KMeans, DBSCAN, AgglomerativeClustering
- Metrics: silhouette_score, inertia
- Demo: Feature inputs → Cluster assignment

### NLP
- Models: Naive Bayes, LSTM, BERT
- Metrics: accuracy, f1, confusion matrix
- Demo: Text input → Sentiment/Classification

### Computer Vision
- Models: CNN, ResNet, VGG
- Metrics: accuracy, top-5 accuracy
- Demo: Image upload → Classification/Detection

### Time Series Forecasting
- Models: Prophet, ARIMA, LSTM
- Metrics: mae, mape, smape
- Demo: Historical data → Future predictions

---

## 📦 File Purposes

| File | Purpose |
|------|---------|
| `src/config.py` | Paths, constants, hyperparameters |
| `src/preprocess.py` | Data cleaning, transformations, splitting |
| `src/train.py` | Model training pipeline |
| `src/evaluate.py` | Metrics calculation and comparison |
| `src/inference.py` | Loading models and making predictions |
| `app/streamlit_app.py` | Interactive web demo |
| `api/main.py` | REST API for predictions |

---

## 🐛 Troubleshooting

### "Model file not found"
→ Train a model first: `python -m src.train`

### "Target column not found"
→ Update `TARGET_COLUMN` in `src/config.py`

### "Feature mismatch"
→ Ensure inference features match training features

### Import errors
→ Run from repo root, not subdirectories

### Streamlit "module not found"
→ Activate venv: `source venv/bin/activate`

---

## 🎓 Learning Path

1. **Understand the template** - Read this file + README.md
2. **Follow a Colab tutorial** - Complete your Udemy project
3. **Export the notebook** - Save to `notebooks/`
4. **Adapt the code** - Map notebook code to `src/` modules
5. **Train and test** - Run `python -m src.train`
6. **Build demo** - Customize `app/streamlit_app.py`
7. **Deploy** - Push to Hugging Face Spaces or Render
8. **Document** - Fill in README.md for portfolio

---

## 📚 Next Steps

1. ✅ Template is set up
2. 📓 Add your Colab notebook to `notebooks/`
3. 📊 Place data in `data/raw/`
4. 🔧 Customize `src/config.py`
5. 🎨 Adapt preprocessing in `src/preprocess.py`
6. 🚂 Run training: `python -m src.train`
7. 🎭 Test demo: `streamlit run app/streamlit_app.py`
8. 🌐 Deploy and add link to README.md

---

**Need help?** Check the TODO comments in each file for customization hints.
