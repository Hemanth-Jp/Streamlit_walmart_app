import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import pickle

# Set page config
st.set_page_config(
    page_title="Walmart Sales Prediction",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state for model storage
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'model_source' not in st.session_state:
    st.session_state.model_source = None

# ============================================================================
# CORE FUNCTIONS FOR MODEL LOADING AND PREDICTION
# ============================================================================

def load_default_model(model_type):
    """Load default model from models/default/ directory"""
    model_path = f"models/default/{model_type}.pkl"
    
    if not os.path.exists(model_path):
        return None, f"Default {model_type} model not found at {model_path}"
    
    try:
        # Load both model types using joblib for consistency
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def load_uploaded_model(uploaded_file, model_type):
    """Load model from uploaded file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Load both model types using joblib for consistency
        model = joblib.load(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return model, None
    except Exception as e:
        # Clean up if error occurs
        try:
            os.unlink(tmp_path)
        except:
            pass
        return None, f"Invalid model file. Please check format or retrain."

def predict_next_4_weeks(model, model_type):
    """Predict next 4 weeks of sales"""
    # Generate dates for next 4 weeks
    today = datetime.now()
    dates = [today + timedelta(weeks=i) for i in range(1, 5)]
    
    try:
        if model_type == "auto_arima":
            predictions = model.predict(n_periods=4)
        elif model_type == "exponential_smoothing":
            predictions = model.forecast(4)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return predictions, dates, None
    except Exception as e:
        return None, None, f"Error generating predictions: {str(e)}"

# ============================================================================
# STREAMLIT APP INTERFACE
# ============================================================================

def main():
    # App title and description
    st.title("🔮 Walmart Sales Prediction")
    st.markdown("""
    This app generates sales forecasts for the next 4 weeks using trained time series models.
    
    **You can:**
    - Use pre-loaded default models (recommended)
    - Upload your own trained models
    - View interactive forecasts
    - Download prediction results
    """)
    
    # Model selection section
    st.header("🤖 Model Selection")
    
    # Tabs for default vs uploaded models
    tab1, tab2 = st.tabs(["Default Models", "Upload Model"])
    
    with tab1:
        st.subheader("Use Default Models")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Auto ARIMA Model", use_container_width=True):
                model, error = load_default_model("auto_arima")
                if error:
                    st.error(error)
                else:
                    st.session_state.current_model = model
                    st.session_state.model_type = "auto_arima"
                    st.session_state.model_source = "Default"
                    st.success("✅ Auto ARIMA model loaded successfully!")
        
        with col2:
            if st.button("Load Exponential Smoothing Model", use_container_width=True):
                model, error = load_default_model("exponential_smoothing")
                if error:
                    st.error(error)
                else:
                    st.session_state.current_model = model
                    st.session_state.model_type = "exponential_smoothing"
                    st.session_state.model_source = "Default"
                    st.success("✅ Exponential Smoothing model loaded successfully!")
    
    with tab2:
        st.subheader("Upload Custom Model")
        
        # Model type selection for upload
        model_type = st.selectbox(
            "Select model type:",
            ["auto_arima", "exponential_smoothing"],
            key="upload_model_type"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload model file (.pkl)", 
            type=["pkl"],
            key="model_uploader"
        )
        
        if uploaded_file:
            if st.button("Load Uploaded Model", use_container_width=True):
                model, error = load_uploaded_model(uploaded_file, model_type)
                if error:
                    st.error(error)
                else:
                    st.session_state.current_model = model
                    st.session_state.model_type = model_type
                    st.session_state.model_source = "Uploaded"
                    st.success(f"✅ {model_type} model loaded successfully!")
    
    # Display current model info
    if st.session_state.current_model is not None:
        st.info(f"**Current Model:** {st.session_state.model_type.replace('_', ' ').title()} ({st.session_state.model_source})")
    else:
        st.warning("No model loaded. Please select a model to make predictions.")
    
    # Prediction section
    st.header("📈 Generate Predictions")
    
    if st.session_state.current_model is not None:
        if st.button("Generate 4-Week Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                predictions, dates, error = predict_next_4_weeks(
                    st.session_state.current_model,
                    st.session_state.model_type
                )
                
                if error:
                    st.error(error)
                else:
                    # Create prediction dataframe
                    prediction_df = pd.DataFrame({
                        'Date': dates,
                        'Predicted_Sales': predictions,
                        'Week': [f"Week {i+1}" for i in range(4)]
                    })
                    
                    # Display results
                    st.subheader("📊 Forecast Results")
                    
                    # Create interactive plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=prediction_df['Date'],
                        y=prediction_df['Predicted_Sales'],
                        mode='lines+markers',
                        name='Predicted Sales',
                        line=dict(color='blue', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.update_layout(
                        title='Sales Forecast for Next 4 Weeks',
                        xaxis_title='Date',
                        yaxis_title='Weekly Sales',
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    # Add grid
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display data table
                    st.subheader("📋 Prediction Values")
                    st.dataframe(prediction_df[['Week', 'Date', 'Predicted_Sales']], use_container_width=True)
                    
                    # Download section
                    st.subheader("💾 Download Results")
                    
                    # Prepare CSV for download
                    csv = prediction_df.to_csv(index=False).encode('utf-8')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="walmart_sales_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Create JSON version
                        json_str = prediction_df.to_json(orient='records', date_format='iso')
                        st.download_button(
                            label="Download as JSON",
                            data=json_str,
                            file_name="walmart_sales_predictions.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Predicted Sales", f"${predictions.sum():,.2f}")
                    with col2:
                        st.metric("Average Weekly Sales", f"${predictions.mean():,.2f}")
                    with col3:
                        st.metric("Highest Week", f"Week {predictions.argmax() + 1}")
                    with col4:
                        st.metric("Growth Rate", f"{((predictions[-1]/predictions[0])-1)*100:.1f}%")
    
    else:
        st.info("👆 Please load a model first to generate predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Walmart Sales Forecasting System © 2025</p>
        <p><small>Built with ❤️ using Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()