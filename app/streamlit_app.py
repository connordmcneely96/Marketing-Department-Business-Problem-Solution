"""
Streamlit Demo Application for Customer Segmentation.

This app allows users to input credit card customer data and receive
cluster assignment predictions with detailed insights.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import predict_cluster, get_all_clusters_info, create_customer_profile
from src.config import NUMERIC_FEATURES, CLUSTER_NAMES, CLUSTER_DESCRIPTIONS

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .cluster-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_safe():
    """Load model and handle errors gracefully."""
    try:
        # Try to load models by calling get_all_clusters_info
        clusters_info = get_all_clusters_info()
        return True, None, clusters_info
    except FileNotFoundError as e:
        return False, str(e), None
    except Exception as e:
        return False, f"Error loading model: {str(e)}", None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.markdown('<div class="main-header">👥 Customer Segmentation System</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p>Predict customer segments for targeted marketing campaigns using K-Means clustering</p>
        <p style='font-size: 0.9rem; color: #666;'>
            Input credit card customer data to identify which behavioral segment they belong to
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model_loaded, error, clusters_info = load_model_safe()

    if not model_loaded:
        st.error(f"❌ {error}")
        st.info("📋 Please train the model first by running: `python -m src.train`")
        st.code("python -m src.train", language="bash")
        return

    # Sidebar - Cluster Information
    with st.sidebar:
        st.header("📊 Customer Segments")
        st.markdown("The model identifies **7 distinct customer segments** based on credit card behavior:")

        for cluster_info in clusters_info:
            with st.expander(f"**{cluster_info['cluster_name']}** (ID: {cluster_info['cluster_id']})"):
                st.write(cluster_info['cluster_description'])

        st.markdown("---")
        st.markdown("### 📖 Feature Descriptions")
        st.markdown("""
        - **Balance**: Current account balance
        - **Purchases**: Total purchase amount
        - **Cash Advance**: Cash advance taken
        - **Credit Limit**: Maximum credit available
        - **Payments**: Total payments made
        - **Tenure**: Months as customer
        """)

    # Main content area
    tabs = st.tabs(["🔍 Single Customer", "📊 Batch Prediction", "ℹ️ About"])

    # ========================================================================
    # TAB 1: Single Customer Prediction
    # ========================================================================
    with tabs[0]:
        st.subheader("Enter Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**💰 Financial Activity**")
            balance = st.number_input(
                "Balance ($)",
                min_value=0.0,
                max_value=100000.0,
                value=1500.0,
                step=100.0,
                help="Current balance in account"
            )
            purchases = st.number_input(
                "Total Purchases ($)",
                min_value=0.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0,
                help="Total purchase amount in 6 months"
            )
            payments = st.number_input(
                "Total Payments ($)",
                min_value=0.0,
                max_value=100000.0,
                value=2000.0,
                step=100.0,
                help="Total payments made"
            )
            credit_limit = st.number_input(
                "Credit Limit ($)",
                min_value=0.0,
                max_value=100000.0,
                value=5000.0,
                step=500.0,
                help="Maximum credit limit"
            )

        with col2:
            st.markdown("**📈 Transaction Patterns**")
            oneoff_purchases = st.number_input(
                "One-off Purchases ($)",
                min_value=0.0,
                max_value=50000.0,
                value=500.0,
                step=50.0,
                help="Maximum purchase in one transaction"
            )
            installments_purchases = st.number_input(
                "Installment Purchases ($)",
                min_value=0.0,
                max_value=50000.0,
                value=500.0,
                step=50.0,
                help="Amount purchased in installments"
            )
            cash_advance = st.number_input(
                "Cash Advance ($)",
                min_value=0.0,
                max_value=50000.0,
                value=300.0,
                step=50.0,
                help="Cash advance taken"
            )
            minimum_payments = st.number_input(
                "Minimum Payments ($)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                step=10.0,
                help="Minimum payments made"
            )

        with col3:
            st.markdown("**📊 Frequency Metrics (0-1)**")
            balance_frequency = st.slider(
                "Balance Update Frequency",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="How often balance is updated (0=never, 1=always)"
            )
            purchases_frequency = st.slider(
                "Purchase Frequency",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="How frequently purchases are made"
            )
            oneoff_purchases_frequency = st.slider(
                "One-off Purchase Frequency",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Frequency of one-time large purchases"
            )
            purchases_installments_frequency = st.slider(
                "Installment Frequency",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Frequency of installment purchases"
            )
            cash_advance_frequency = st.slider(
                "Cash Advance Frequency",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Frequency of cash advances"
            )
            prc_full_payment = st.slider(
                "% Full Payment",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Percentage of full payment paid"
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            cash_advance_trx = st.number_input(
                "Cash Advance Transactions",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
                help="Number of cash advance transactions"
            )
        with col5:
            purchases_trx = st.number_input(
                "Purchase Transactions",
                min_value=0,
                max_value=200,
                value=30,
                step=1,
                help="Number of purchase transactions"
            )
        with col6:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=1,
                max_value=24,
                value=12,
                step=1,
                help="Number of months as customer"
            )

        # Predict button
        if st.button("🔮 Predict Customer Segment", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer profile..."):
                try:
                    # Create customer profile
                    customer = create_customer_profile(
                        balance=balance,
                        balance_frequency=balance_frequency,
                        purchases=purchases,
                        oneoff_purchases=oneoff_purchases,
                        installments_purchases=installments_purchases,
                        cash_advance=cash_advance,
                        purchases_frequency=purchases_frequency,
                        oneoff_purchases_frequency=oneoff_purchases_frequency,
                        purchases_installments_frequency=purchases_installments_frequency,
                        cash_advance_frequency=cash_advance_frequency,
                        cash_advance_trx=cash_advance_trx,
                        purchases_trx=purchases_trx,
                        credit_limit=credit_limit,
                        payments=payments,
                        minimum_payments=minimum_payments,
                        prc_full_payment=prc_full_payment,
                        tenure=tenure,
                    )

                    # Predict
                    result = predict_cluster(customer, return_details=True)

                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")

                    # Main result card
                    st.markdown(f"""
                    <div class="cluster-card">
                        <h2 style='margin: 0; color: #1f77b4;'>{result['cluster_name']}</h2>
                        <p style='font-size: 1.1rem; color: #333; margin-top: 0.5rem;'>
                            {result['cluster_description']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Cluster ID",
                            result['cluster_id'],
                            delta=None
                        )
                    with col2:
                        st.metric(
                            "Confidence Score",
                            f"{result['confidence_score']:.1%}",
                            delta=None
                        )
                    with col3:
                        st.metric(
                            "Distance to Center",
                            f"{result['distance_to_center']:.2f}",
                            delta=None,
                            help="Lower is better (closer to cluster center)"
                        )

                    # Alternative clusters
                    with st.expander("🔄 Alternative Segments (Next Best Matches)"):
                        for i, alt_cluster in enumerate(result['closest_clusters'][:3], 1):
                            st.write(f"**{i}. {alt_cluster['cluster_name']}** - Distance: {alt_cluster['distance']:.2f}")

                    # Marketing recommendations
                    st.markdown("### 💼 Marketing Recommendations")
                    recommendations = {
                        0: "Focus on rewards programs and cashback offers. These customers value convenience.",
                        1: "Offer balance transfer promotions and debt consolidation services.",
                        2: "Provide premium services, concierge benefits, and exclusive rewards.",
                        3: "Re-engagement campaigns with special incentives to increase card usage.",
                        4: "Financial wellness programs and responsible credit education.",
                        5: "Flexible payment plans and installment-friendly offers.",
                        6: "Balanced rewards program with mix of cashback and points.",
                    }
                    st.info(recommendations.get(result['cluster_id'], "Standard marketing approach recommended."))

                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.exception(e)

    # ========================================================================
    # TAB 2: Batch Prediction
    # ========================================================================
    with tabs[1]:
        st.subheader("Upload Customer Data for Batch Prediction")

        st.markdown("""
        Upload a CSV file with customer data. The file should contain the following columns:
        - BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES
        - CASH_ADVANCE, PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY
        - PURCHASES_INSTALLMENTS_FREQUENCY, CASH_ADVANCE_FREQUENCY
        - CASH_ADVANCE_TRX, PURCHASES_TRX, CREDIT_LIMIT, PAYMENTS
        - MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Preview:**")
                st.dataframe(df.head())

                if st.button("🔮 Predict All Segments", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        from src.inference import predict_batch
                        results = predict_batch(df, return_details=True)

                        # Add results to dataframe
                        df['Predicted_Cluster'] = [r['cluster_id'] for r in results]
                        df['Cluster_Name'] = [r['cluster_name'] for r in results]
                        df['Confidence'] = [r['confidence_score'] for r in results]

                        st.success(f"✅ Predicted segments for {len(df)} customers!")
                        st.dataframe(df[['Predicted_Cluster', 'Cluster_Name', 'Confidence']])

                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="customer_segments.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # ========================================================================
    # TAB 3: About
    # ========================================================================
    with tabs[2]:
        st.subheader("About This Application")

        st.markdown("""
        ### 🎯 Project Overview

        This customer segmentation system uses **unsupervised machine learning** (K-Means clustering)
        to identify distinct behavioral groups among credit card customers. The goal is to enable
        **targeted marketing campaigns** tailored to each segment's characteristics.

        ### 📊 The Data

        - **Source**: Credit card customer data over 6 months
        - **Features**: 17 behavioral and transactional metrics
        - **Customers**: Analyzed thousands of customer profiles

        ### 🤖 The Model

        - **Algorithm**: K-Means Clustering
        - **Clusters**: 7 distinct customer segments
        - **Dimensionality Reduction**: PCA for visualization
        - **Optional**: Autoencoder for advanced feature engineering

        ### 🎯 Business Impact

        By segmenting customers, marketing teams can:
        - Design targeted campaigns for each segment
        - Optimize marketing spend and ROI
        - Improve customer engagement and retention
        - Personalize offers and communication

        ### 🛠️ Tech Stack

        - **Python** 3.9+
        - **Scikit-learn** - K-Means, PCA
        - **Pandas / NumPy** - Data processing
        - **Streamlit** - Web interface
        - **TensorFlow / Keras** - Autoencoder (optional)
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Built with ❤️ using Python & Streamlit | Customer Segmentation ML Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
