---
title: Customer Segmentation
emoji: 👥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app/streamlit_app.py
pinned: false
---

# Customer Segmentation - Credit Card Marketing

🎯 **ML-powered customer segmentation for targeted marketing campaigns**

This Streamlit app uses **K-Means clustering** to segment credit card customers into 7 distinct behavioral groups, enabling banks to design targeted marketing strategies.

## 🚀 Features

- **Single Customer Prediction**: Input 17 credit card features and get instant cluster assignment
- **Confidence Scoring**: Distance-based confidence metrics for predictions
- **Marketing Recommendations**: Tailored strategies for each customer segment
- **Batch Predictions**: Upload CSV files for bulk customer segmentation
- **7 Customer Segments**: Transactors, Revolvers, VIP/Prime, Low Activity, and more

## 📊 Model Details

- **Algorithm**: K-Means Clustering (scikit-learn)
- **Clusters**: 7 distinct customer segments
- **Features**: 17 credit card behavioral metrics
- **Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score

## 🛠️ Setup

### Training the Model

Before using the app, you need to train the model:

1. Add your credit card dataset to `data/raw/CC_GENERAL.csv`
2. Train the model:
   ```bash
   python -m src.train
   ```
3. Models will be saved to the `models/` directory

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
├── app/
│   └── streamlit_app.py    # Main Streamlit application
├── src/
│   ├── config.py           # Configuration and parameters
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Evaluation metrics
│   └── inference.py        # Prediction functions
├── models/                 # Trained models (after training)
├── data/                   # Dataset directory
└── notebooks/              # Jupyter notebooks
```

## 🎓 About

This project demonstrates a complete ML workflow:
- Data preprocessing with StandardScaler and imputation
- K-Means clustering for unsupervised learning
- Production-ready inference with caching
- Professional Streamlit UI with real-time predictions

**Tech Stack**: Python, scikit-learn, Pandas, Streamlit, Matplotlib

---

💡 **Note**: Models need to be trained before the app can make predictions. Upload your dataset and follow the training instructions above.
