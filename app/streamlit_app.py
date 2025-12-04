"""
Streamlit Demo Application for ML Project.

This is a template that should be customized based on your specific project type:
- Classification: Input features -> Predict class
- Regression: Input features -> Predict value
- NLP: Text input -> Sentiment/Classification
- Computer Vision: Image upload -> Classification/Detection
- Time Series: Historical data -> Forecast

TODO: Customize this file based on your project requirements.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import (
    predict_classification,
    predict_regression,
    get_model_info,
    load_model,
)
from src.config import MODEL_FILE


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="ML Model Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_safe():
    """Load model and handle errors gracefully."""
    try:
        bundle = load_model()
        return bundle, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def display_model_info(metadata: dict):
    """Display model information in sidebar."""
    st.sidebar.header("📊 Model Info")
    st.sidebar.write(f"**Model:** {metadata.get('model_name', 'Unknown')}")
    st.sidebar.write(f"**Type:** {metadata.get('model_type', 'Unknown')}")
    st.sidebar.write(f"**Features:** {metadata.get('n_features', 'Unknown')}")
    
    if "metrics" in metadata:
        st.sidebar.write("**Performance:**")
        for metric, value in metadata["metrics"].items():
            if value is not None:
                st.sidebar.write(f"  - {metric}: {value:.4f}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and description
    st.title("🤖 ML Model Demo")
    st.markdown("""
    This is a demo application for making predictions using a trained machine learning model.
    
    **TODO:** Customize this app based on your specific project:
    - Update the title and description
    - Add appropriate input controls
    - Customize the prediction display
    """)
    
    # Load model
    bundle, error = load_model_safe()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please train a model first by running: `python -m src.train`")
        return
    
    metadata = bundle.get("metadata", {})
    model_type = metadata.get("model_type", "classification")
    
    # Display model info in sidebar
    display_model_info(metadata)
    
    # ========================================================================
    # TABULAR CLASSIFICATION/REGRESSION DEMO
    # ========================================================================
    
    if model_type in ["classification", "regression"]:
        st.header("📊 Make a Prediction")
        
        # Get feature names
        feature_names = metadata.get("feature_names", [])
        
        if not feature_names:
            st.warning("⚠️ Feature names not found in model metadata. Using generic names.")
            n_features = metadata.get("n_features", 5)
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Create input form
        st.subheader("Input Features")
        
        # Option 1: Manual input
        with st.expander("📝 Enter features manually", expanded=True):
            cols = st.columns(3)
            features = {}
            
            for i, feature_name in enumerate(feature_names):
                with cols[i % 3]:
                    # TODO: Customize input types based on your features
                    # Use st.selectbox for categorical, st.number_input for numeric
                    features[feature_name] = st.number_input(
                        feature_name,
                        value=0.0,
                        help=f"Enter value for {feature_name}",
                    )
        
        # Option 2: Upload CSV (for batch predictions)
        with st.expander("📁 Upload CSV file for batch predictions"):
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:")
                st.dataframe(df.head())
                
                # TODO: Implement batch prediction
                st.info("Batch prediction feature coming soon!")
        
        # Make prediction button
        if st.button("🔮 Make Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    if model_type == "classification":
                        result = predict_classification(features, return_proba=True)
                        
                        # Display prediction
                        st.success(f"### Predicted Class: **{result['prediction']}**")
                        
                        # Display probabilities if available
                        if "probabilities" in result:
                            st.write(f"**Confidence:** {result['confidence']:.2%}")
                            
                            # Show probability distribution
                            st.write("**Class Probabilities:**")
                            prob_df = pd.DataFrame({
                                "Class": range(len(result['probabilities'])),
                                "Probability": result['probabilities']
                            })
                            st.bar_chart(prob_df.set_index("Class"))
                    
                    elif model_type == "regression":
                        prediction = predict_regression(features)
                        st.success(f"### Predicted Value: **{prediction:.4f}**")
                    
                    # Display input summary
                    with st.expander("📋 Input Summary"):
                        st.json(features)
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.exception(e)
    
    # ========================================================================
    # NLP DEMO
    # ========================================================================
    
    elif model_type == "nlp":
        st.header("📝 Text Analysis")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here...",
        )
        
        if st.button("🔮 Analyze", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # TODO: Implement text prediction
                    st.info("NLP prediction not yet implemented. See src.inference.predict_from_text()")
            else:
                st.warning("Please enter some text to analyze.")
    
    # ========================================================================
    # COMPUTER VISION DEMO
    # ========================================================================
    
    elif model_type == "computer_vision":
        st.header("🖼️ Image Analysis")
        
        uploaded_image = st.file_uploader(
            "Upload an image:",
            type=["jpg", "jpeg", "png"],
        )
        
        if uploaded_image is not None:
            # Display image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔮 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # TODO: Implement image prediction
                    st.info("Image prediction not yet implemented. See src.inference.predict_from_image()")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit | Deployed on Hugging Face Spaces / Render</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
