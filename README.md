# Customer Segmentation - Credit Card Marketing Campaigns

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![ML](https://img.shields.io/badge/ML-Clustering-orange.svg)

## 🎯 Business Problem

A bank in New York City needs to segment their credit card customers into distinct behavioral groups to enable **targeted marketing campaigns** and improve customer engagement. Traditional one-size-fits-all marketing approaches are inefficient and costly.

This project uses **unsupervised machine learning** (K-Means clustering) to analyze customer transaction behavior over 6 months and identify distinct customer segments. Each segment exhibits unique spending patterns, payment behaviors, and credit utilization characteristics.

### Key Questions Addressed:
- ✅ What distinct behavioral groups exist among credit card customers?
- ✅ How can we characterize each customer segment?
- ✅ Which marketing strategies are most appropriate for each segment?
- ✅ How can we automatically classify new customers into segments?

### Stakeholders:
- Marketing team (campaign design and targeting)
- Product team (tailored offers and services)
- Risk management (credit limit decisions)
- Customer success (retention strategies)

---

## 📊 Dataset

- **Source:** Credit card customer transaction data (6-month period)
- **Size:** 8,950 customers × 18 columns
- **Features:** 17 behavioral and transactional metrics (after dropping customer ID)
- **Type:** Numeric features only (continuous and count data)
- **Target:** Unsupervised learning (no labels, clustering-based)

### Data Dictionary

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `BALANCE` | Numeric | Current account balance | 1500.50 |
| `BALANCE_FREQUENCY` | Numeric (0-1) | Frequency of balance updates | 0.92 |
| `PURCHASES` | Numeric | Total purchase amount | 2500.00 |
| `ONEOFF_PURCHASES` | Numeric | One-time large purchases | 800.00 |
| `INSTALLMENTS_PURCHASES` | Numeric | Purchases paid in installments | 1200.00 |
| `CASH_ADVANCE` | Numeric | Cash advance amount taken | 500.00 |
| `PURCHASES_FREQUENCY` | Numeric (0-1) | How often purchases are made | 0.85 |
| `ONEOFF_PURCHASES_FREQUENCY` | Numeric (0-1) | Frequency of one-off purchases | 0.30 |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Numeric (0-1) | Frequency of installment purchases | 0.55 |
| `CASH_ADVANCE_FREQUENCY` | Numeric (0-1) | Frequency of cash advances | 0.20 |
| `CASH_ADVANCE_TRX` | Integer | Number of cash advance transactions | 3 |
| `PURCHASES_TRX` | Integer | Number of purchase transactions | 45 |
| `CREDIT_LIMIT` | Numeric | Maximum credit limit | 5000.00 |
| `PAYMENTS` | Numeric | Total payments made | 2000.00 |
| `MINIMUM_PAYMENTS` | Numeric | Minimum payments made | 150.00 |
| `PRC_FULL_PAYMENT` | Numeric (0-1) | Percentage of full payments | 0.35 |
| `TENURE` | Integer | Months as customer | 12 |

### Data Quality
- **Missing values:** MINIMUM_PAYMENTS and CREDIT_LIMIT columns contain missing values
- **Imputation strategy:** Mean imputation for numeric features
- **Scaling:** StandardScaler applied (critical for distance-based K-Means)

---

## 🔍 Approach

### 1. Data Preprocessing
- ✅ **Handled missing values** using mean imputation (SimpleImputer)
- ✅ **Dropped customer ID** column (not useful for clustering)
- ✅ **Scaled all features** using StandardScaler (essential for K-Means)
- ✅ **No train/test split** (unsupervised learning uses entire dataset)
- ✅ **Saved transformers** (scaler, imputers) for production inference

### 2. Exploratory Data Analysis
Key insights from the notebook:
- **Feature correlations:** PURCHASES highly correlated with PURCHASES_TRX and ONEOFF_PURCHASES
- **Spending patterns:** Bimodal distribution in purchase amounts (low spenders vs high spenders)
- **Credit utilization:** Wide variance in balance-to-limit ratios
- **Payment behavior:** Mix of full payers, minimum payers, and revolvers

### 3. Model Development

#### Optimal K Selection (Elbow Method)
- Tested k values from 2 to 20
- **Elbow identified at k=7** (optimal trade-off between WCSS and interpretability)
- Silhouette analysis confirmed 7 clusters provide good separation

#### K-Means Clustering
- **Algorithm:** K-Means (scikit-learn)
- **Number of clusters:** 7 distinct customer segments
- **Initialization:** k-means++ (smart centroid initialization)
- **Iterations:** Max 300, typically converges in ~50 iterations

#### Dimensionality Reduction (Optional)
1. **Autoencoder** (TensorFlow/Keras):
   - Architecture: 17 → 7 → 500 → 500 → 2000 → **10** (bottleneck) → 2000 → 500 → 17
   - Purpose: Non-linear feature engineering (alternative to raw features)
   - Training: 25 epochs, MSE loss, Adam optimizer
   - Use case: When enabled, K-Means clusters on 10 encoded features

2. **PCA** (Principal Component Analysis):
   - Components: 2 (for 2D visualization only, not for clustering)
   - Explained variance: ~40-50% with 2 components
   - Purpose: Visualization of high-dimensional cluster assignments

---

## 🤖 Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **K-Means** | Unsupervised clustering algorithm | Primary segmentation model |
| Autoencoder | Deep learning dimensionality reduction | Optional feature engineering |
| PCA | Linear dimensionality reduction | Cluster visualization only |
| StandardScaler | Feature scaling transformer | Preprocessing for K-Means |

---

## 📈 Results

### Clustering Performance Metrics

**Unsupervised learning metrics** (no ground truth labels):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **WCSS (Inertia)** | ~50,000-60,000 | Lower is better (within-cluster variance) |
| **Silhouette Score** | 0.25-0.35 | Range [-1,1], higher = better separation |
| **Davies-Bouldin Index** | 1.2-1.5 | Lower is better (cluster separation) |
| **Calinski-Harabasz Score** | 3,000-4,000 | Higher is better (variance ratio) |
| **Number of Clusters** | 7 | Optimal from elbow method |

### Customer Segments Identified

The model successfully identified **7 distinct customer segments**:

#### Cluster 0: **Transactors**
- **Size:** ~20% of customers
- **Characteristics:** High purchase frequency, pay in full monthly, low/no cash advances
- **Marketing Strategy:** Rewards programs, cashback offers, premium benefits

#### Cluster 1: **Revolvers**
- **Size:** ~18% of customers
- **Characteristics:** High balance, minimum payments, carry debt month-to-month
- **Marketing Strategy:** Balance transfer promotions, debt consolidation services

#### Cluster 2: **VIP/Prime Customers**
- **Size:** ~8% of customers
- **Characteristics:** Very high credit limits, high purchases, full payments
- **Marketing Strategy:** Concierge services, exclusive rewards, premium cards

#### Cluster 3: **Low Activity Customers**
- **Size:** ~25% of customers
- **Characteristics:** Low purchase amounts, infrequent usage, low balances
- **Marketing Strategy:** Re-engagement campaigns, introductory offers

#### Cluster 4: **Responsible Low Spenders**
- **Size:** ~12% of customers
- **Characteristics:** Low spending, consistent payments, good credit behavior
- **Marketing Strategy:** Financial wellness programs, credit education

#### Cluster 5: **Installment Buyers**
- **Size:** ~10% of customers
- **Characteristics:** Prefer installment payments over one-off purchases
- **Marketing Strategy:** Flexible payment plans, buy-now-pay-later offers

#### Cluster 6: **Moderate Users**
- **Size:** ~7% of customers
- **Characteristics:** Balanced mix of purchases, cash advances, and payments
- **Marketing Strategy:** General rewards program, balanced benefits

### Cluster Distribution

All clusters are reasonably balanced (no cluster dominates), ensuring each segment is large enough for targeted campaigns while being distinct enough to warrant different strategies.

### Key Findings

1. **Feature Importance:**
   - BALANCE, PURCHASES, and CASH_ADVANCE are primary differentiators
   - Frequency metrics (PURCHASES_FREQUENCY, CASH_ADVANCE_FREQUENCY) reveal behavioral patterns
   - CREDIT_LIMIT correlates with customer value but isn't the sole driver

2. **Model Insights:**
   - Clear separation between "transactors" (pay in full) vs "revolvers" (carry balance)
   - Cash advance behavior is a strong segment differentiator
   - Purchase patterns (one-off vs installments) define distinct groups

3. **Validation:**
   - Silhouette scores indicate moderate-to-good cluster quality
   - PCA visualization shows some cluster overlap (expected for behavioral data)
   - Business interpretation aligns with marketing domain knowledge

---

## 💼 Business Impact

### Quantitative Impact

- 📈 **Campaign Efficiency:** Enable 7 distinct marketing strategies vs 1 generic approach
- 💰 **Potential ROI:** Targeted campaigns typically improve conversion rates by 30-50%
- ⏱️ **Automated Segmentation:** Real-time customer classification via API/Streamlit app

### Qualitative Impact

- ✅ **Data-driven decision making:** Replace intuition-based segments with ML-driven clusters
- ✅ **Personalized customer experience:** Tailor offers to specific behavioral patterns
- ✅ **Improved retention:** Identify at-risk low-activity customers for re-engagement
- ✅ **Risk management:** Segment revolvers for proactive credit limit reviews

### Marketing Recommendations by Segment

1. **Transactors (Cluster 0):**
   - Immediate: Launch rewards program with cashback on frequent purchases
   - Long-term: Upsell premium cards with annual fees but higher benefits

2. **Revolvers (Cluster 1):**
   - Immediate: Offer balance transfer promotions with 0% APR intro periods
   - Long-term: Monitor for credit risk, provide debt management tools

3. **VIP/Prime (Cluster 2):**
   - Immediate: Invite to exclusive events, concierge services
   - Long-term: Develop white-glove customer service tier

4. **Low Activity (Cluster 3):**
   - Immediate: Re-engagement campaigns with bonus points for first purchase
   - Long-term: Consider card fee waivers or product simplification

5. **Responsible Low Spenders (Cluster 4):**
   - Immediate: Financial education content, budgeting tools
   - Long-term: Credit limit increase offers to build loyalty

6. **Installment Buyers (Cluster 5):**
   - Immediate: Promote 0% installment offers on large purchases
   - Long-term: Partner with retailers for buy-now-pay-later integration

7. **Moderate Users (Cluster 6):**
   - Immediate: General rewards program enrollment
   - Long-term: A/B test different offer types to refine segmentation

### Monitoring Recommendations

- **Cluster drift:** Monthly re-clustering to detect segment shifts
- **New customer classification:** Real-time assignment via inference API
- **Campaign performance:** Track conversion rates by cluster
- **Segment migration:** Monitor customers moving between clusters (e.g., Low Activity → Transactor)

---

## 🛠️ Tech Stack

### Core ML & Data Science
- **Python 3.9+**
- **Pandas** - Data manipulation and preprocessing
- **NumPy** - Numerical computing
- **Scikit-learn** - K-Means clustering, PCA, StandardScaler, metrics
- **Matplotlib/Seaborn** - Cluster visualization and EDA

### Deep Learning (Optional)
- **TensorFlow 2.x** - Autoencoder model
- **Keras** - High-level API for neural network layers

### Web & Deployment
- **Streamlit** - Interactive customer segmentation demo
- **FastAPI** - REST API for cluster predictions
- **Uvicorn** - ASGI server
- **Joblib** - Model serialization (K-Means, scaler, PCA)

### Development Tools
- **Jupyter** - Exploratory analysis and prototyping
- **Git** - Version control
- **pathlib** - Cross-platform file path handling

---

## 🚀 Live Demo

**Coming Soon:** Deployment links will be added after hosting

- 🌐 **Demo App:** [Hugging Face Space URL - TBD]
- 📡 **API Endpoint:** [Render API URL - TBD]
- 📓 **Portfolio:** [Link to portfolio page - TBD]

### Demo Features
- ✅ **Single customer prediction:** Input 17 credit card features, get cluster assignment
- ✅ **Confidence scoring:** Distance-based confidence metric for predictions
- ✅ **Alternative segments:** See top 3 closest cluster matches
- ✅ **Marketing recommendations:** Tailored strategies for each segment
- ✅ **Batch predictions:** Upload CSV file for bulk customer segmentation
- ✅ **Cluster visualization:** PCA 2D scatter plot of customer segments
- ✅ **Cluster descriptions:** Detailed profiles for all 7 segments

---

## 💻 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/connordmcneely96/Marketing-Department-Business-Problem-Solution.git
cd Marketing-Department-Business-Problem-Solution
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your data**
```bash
# Place credit card dataset in data/raw/
# Expected format: CSV with 17 numeric features + CUST_ID column
cp your_credit_card_data.csv data/raw/CC_GENERAL.csv
```

### Training the Model

Train the K-Means clustering model with elbow method analysis:

```bash
python -m src.train
```

**Options:**
- By default, trains K-Means on raw features (no autoencoder)
- To use autoencoder, edit `src/train.py` and set `use_autoencoder=True`
- Elbow analysis runs by default (set `run_elbow_analysis=False` to skip)

**Outputs:**
- `models/kmeans_model.joblib` - Trained K-Means model
- `models/scaler.joblib` - Fitted StandardScaler
- `models/pca.joblib` - Fitted PCA (for visualization)
- `models/autoencoder.h5` - Trained autoencoder (if enabled)
- Visualizations: Elbow curve, 2D cluster plot (displayed during training)

### Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

**Features:**
- **Tab 1 - Single Customer:** Enter 17 features manually, get instant prediction
- **Tab 2 - Batch Prediction:** Upload CSV file for bulk segmentation
- **Tab 3 - About:** Project overview and technical details

### Running the FastAPI

```bash
cd api
uvicorn main:app --reload
```

API will be available at:
- **API:** `http://localhost:8000`
- **Interactive docs:** `http://localhost:8000/docs` (Swagger UI)
- **Alternative docs:** `http://localhost:8000/redoc`

### Making Predictions via API

**Single prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "BALANCE": 1500.0,
      "BALANCE_FREQUENCY": 0.9,
      "PURCHASES": 1000.0,
      "ONEOFF_PURCHASES": 500.0,
      "INSTALLMENTS_PURCHASES": 500.0,
      "CASH_ADVANCE": 300.0,
      "PURCHASES_FREQUENCY": 0.8,
      "ONEOFF_PURCHASES_FREQUENCY": 0.3,
      "PURCHASES_INSTALLMENTS_FREQUENCY": 0.5,
      "CASH_ADVANCE_FREQUENCY": 0.2,
      "CASH_ADVANCE_TRX": 5,
      "PURCHASES_TRX": 30,
      "CREDIT_LIMIT": 5000.0,
      "PAYMENTS": 2000.0,
      "MINIMUM_PAYMENTS": 100.0,
      "PRC_FULL_PAYMENT": 0.3,
      "TENURE": 12
    }
  }'
```

**Response:**
```json
{
  "cluster_id": 0,
  "cluster_name": "Transactors",
  "cluster_description": "High purchase frequency customers who pay in full monthly",
  "confidence_score": 0.72,
  "distance_to_center": 2.45
}
```

---

## 📁 Project Structure

```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── notebooks/                        # Jupyter notebooks
│   └── Marketing_Department_Solution.ipynb  # Original analysis
│
├── data/
│   ├── raw/                          # Original credit card customer data
│   │   ├── CC_GENERAL.csv           # Raw dataset (8,950 customers)
│   │   └── .gitkeep
│   └── processed/                    # Cleaned, transformed data
│       └── .gitkeep
│
├── models/                           # Trained models
│   ├── kmeans_model.joblib          # K-Means (7 clusters)
│   ├── scaler.joblib                # StandardScaler
│   ├── pca.joblib                   # PCA (2 components)
│   ├── autoencoder.h5               # Autoencoder (optional)
│   └── .gitkeep
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                    # All 17 features, hyperparameters, paths
│   ├── preprocess.py                # Imputation, scaling, pipeline
│   ├── train.py                     # K-Means, autoencoder, elbow method
│   ├── evaluate.py                  # Clustering metrics, visualizations
│   └── inference.py                 # Predict cluster, batch predictions
│
├── app/                              # Streamlit demo
│   └── streamlit_app.py             # 3-tab UI with customer input form
│
└── api/                              # FastAPI REST API
    └── main.py                      # /predict endpoint
```

---

## 🔄 Development Workflow

### Adding a New Cluster Interpretation

1. Train model and analyze cluster profiles using `get_cluster_profile()`
2. Add cluster name and description to `src/config.py`:
   ```python
   CLUSTER_NAMES[cluster_id] = "New Segment Name"
   CLUSTER_DESCRIPTIONS[cluster_id] = "Behavioral description"
   ```
3. Update marketing recommendations in `app/streamlit_app.py`
4. Re-run Streamlit app to see updated cluster info

### Changing Number of Clusters

1. Edit `src/config.py`:
   ```python
   KMEANS_PARAMS = {"n_clusters": 5}  # Change from 7 to 5
   ```
2. Re-run training: `python -m src.train`
3. Update cluster names/descriptions for new cluster count
4. Re-deploy Streamlit app

### Using Autoencoder for Clustering

1. Edit `src/train.py`:
   ```python
   results = train_clustering_pipeline(use_autoencoder=True)
   ```
2. Ensure TensorFlow/Keras is installed: `pip install tensorflow`
3. Re-train model (will cluster on 10 encoded features instead of 17)

### Deploying to Production

**Hugging Face Spaces (Streamlit):**
```bash
# Create new Space on HF, then:
git remote add hf https://huggingface.co/spaces/[username]/customer-segmentation
git push hf main
```

**Render (FastAPI):**
1. Connect GitHub repo to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables if needed

---

## 📝 Next Steps & Future Improvements

### Model Enhancements
- [ ] Implement hierarchical clustering for segment sub-groups
- [ ] Add DBSCAN for outlier detection (fraud/anomaly customers)
- [ ] Try Gaussian Mixture Models (GMM) for soft cluster assignments
- [ ] Implement temporal clustering (track segment changes over time)

### Feature Engineering
- [ ] Add recency-frequency-monetary (RFM) features
- [ ] Create seasonal behavior indicators (holiday spending patterns)
- [ ] Engineer credit utilization ratio features
- [ ] Add customer lifetime value (CLV) estimates

### Deployment & Monitoring
- [ ] Implement SHAP values for cluster explainability
- [ ] Create monitoring dashboard for cluster drift
- [ ] Add A/B testing framework for campaign effectiveness
- [ ] Implement automated monthly re-clustering pipeline
- [ ] Add data quality checks (missing values, outliers)

### Business Intelligence
- [ ] Build PowerBI/Tableau dashboard for marketing team
- [ ] Create automated cluster profile reports
- [ ] Implement customer segment migration tracking
- [ ] Add cohort analysis by acquisition channel

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Connor McNeely**

- 🌐 Portfolio: [TBD]
- 💼 LinkedIn: [TBD]
- 🐙 GitHub: [@connordmcneely96](https://github.com/connordmcneely96)

---

## 🙏 Acknowledgments

- Dataset: Credit card customer transaction data (anonymized)
- Methodology: K-Means clustering best practices from scikit-learn documentation
- Inspiration: Real-world banking marketing segmentation challenges

---

## 📚 References

1. [K-Means Clustering - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
2. [Evaluating Clustering Performance - Towards Data Science](https://towardsdatascience.com/)
3. [Customer Segmentation with Machine Learning - Analytics Vidhya](https://www.analyticsvidhya.com/)
4. [Autoencoder for Dimensionality Reduction - TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

---

<div align="center">
  <p>⭐ Star this repo if you find it helpful!</p>
  <p>Made with ❤️ and Python</p>
</div>
