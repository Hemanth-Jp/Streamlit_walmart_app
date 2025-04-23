import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from model import RandomForestModel, TimeSeriesModel
from utils import preprocess_data, create_visualizations, evaluate_model

# Set page configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Helper function to reset app state
def reset_app():
    st.session_state.data_uploaded = False
    st.session_state.data_processed = False
    st.session_state.model_trained = False
    st.session_state.predictions_made = False
    if 'df' in st.session_state:
        del st.session_state.df
    if 'rf_model' in st.session_state:
        del st.session_state.rf_model
    if 'ts_model' in st.session_state:
        del st.session_state.ts_model
    if 'X_test' in st.session_state:
        del st.session_state.X_test
    if 'y_test' in st.session_state:
        del st.session_state.y_test
    if 'test_data' in st.session_state:
        del st.session_state.test_data

# App header
st.title("Walmart Sales Forecasting Dashboard")
st.markdown("""
This app demonstrates the analysis and forecasting of Walmart sales data using 
machine learning and time series techniques.
""")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select a page",
    ["Data Upload", "Data Exploration", "Model Training", "Predictions & Evaluation"]
)

# Reset button in sidebar
if st.sidebar.button("Reset App"):
    reset_app()
    st.sidebar.success("App has been reset!")

# Data Upload Page
if app_mode == "Data Upload":
    st.header("Data Upload")
    st.markdown("""
    Please upload the three required CSV files:
    1. **stores.csv** - Information about the stores
    2. **train.csv** - Weekly sales data for training
    3. **features.csv** - Additional features like promotions, holidays, etc.
    """)
    
    # File uploaders
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stores_file = st.file_uploader("Upload stores.csv", type=['csv'])
    with col2:
        train_file = st.file_uploader("Upload train.csv", type=['csv'])
    with col3:
        features_file = st.file_uploader("Upload features.csv", type=['csv'])
    
    # Process uploaded files
    if stores_file and train_file and features_file:
        try:
            # Read data
            df_store = pd.read_csv(stores_file)
            df_train = pd.read_csv(train_file)
            df_features = pd.read_csv(features_file)
            
            # Display summary
            st.success("All files uploaded successfully!")
            st.write("Summary of uploaded data:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stores", df_store['Store'].nunique())
            with col2:
                st.metric("Departments", df_train['Dept'].nunique())
            with col3:
                st.metric("Date Range", f"{df_train['Date'].min()} to {df_train['Date'].max()}")
            
            # Store dataframes in session state
            st.session_state.df_store = df_store
            st.session_state.df_train = df_train
            st.session_state.df_features = df_features
            st.session_state.data_uploaded = True
            
            # Sample data display
            with st.expander("View Sample Data"):
                tab1, tab2, tab3 = st.tabs(["Stores", "Sales", "Features"])
                with tab1:
                    st.dataframe(df_store.head())
                with tab2:
                    st.dataframe(df_train.head())
                with tab3:
                    st.dataframe(df_features.head())
            
            # Process data button
            if st.button("Process Data"):
                with st.spinner("Processing data..."):
                    # Merge datasets and clean
                    st.session_state.df = preprocess_data(
                        df_store, df_train, df_features
                    )
                    st.session_state.data_processed = True
                st.success("Data processing complete!")
                st.balloons()
        
        except Exception as e:
            st.error(f"Error processing files: {e}")
    
    else:
        st.info("Please upload all three required files.")

# Data Exploration Page
elif app_mode == "Data Exploration":
    st.header("Data Exploration")
    
    if not st.session_state.data_uploaded:
        st.warning("Please upload data files first!")
        st.stop()
    
    if not st.session_state.data_processed:
        st.warning("Please process the data first!")
        
        if st.button("Process Data Now"):
            with st.spinner("Processing data..."):
                st.session_state.df = preprocess_data(
                    st.session_state.df_store,
                    st.session_state.df_train,
                    st.session_state.df_features
                )
                st.session_state.data_processed = True
            st.success("Data processing complete!")
    
    if st.session_state.data_processed:
        # Create tabs for different visualization categories
        tab1, tab2, tab3, tab4 = st.tabs(["Store Analysis", "Holiday Impact", "Time Analysis", "External Factors"])
        
        with tab1:
            st.subheader("Store Type and Department Analysis")
            fig1, fig2, fig3 = create_visualizations(st.session_state.df, "store_analysis")
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.pyplot(fig3)
        
        with tab2:
            st.subheader("Holiday Impact Analysis")
            fig1, fig2 = create_visualizations(st.session_state.df, "holiday_analysis")
            st.pyplot(fig1)
            st.pyplot(fig2)
        
        with tab3:
            st.subheader("Time Analysis")
            fig1, fig2 = create_visualizations(st.session_state.df, "time_analysis")
            st.pyplot(fig1)
            st.pyplot(fig2)
        
        with tab4:
            st.subheader("External Factors Analysis")
            fig = create_visualizations(st.session_state.df, "external_factors")
            st.pyplot(fig)

# Model Training Page
elif app_mode == "Model Training":
    st.header("Model Training")
    
    if not st.session_state.data_processed:
        st.warning("Please process the data first!")
        st.stop()
    
    st.write("Select models to train:")
    col1, col2 = st.columns(2)
    
    with col1:
        train_rf = st.checkbox("Random Forest", value=True)
        rf_params = {}
        if train_rf:
            st.subheader("Random Forest Parameters")
            rf_params['n_estimators'] = st.slider("Number of Trees", 10, 100, 50)
            rf_params['max_depth'] = st.slider("Max Depth", 10, 50, 35)
            feature_options = ['auto', 'sqrt', 'log2']
            rf_params['max_features'] = st.selectbox("Max Features", feature_options, 1)
    
    with col2:
        train_ts = st.checkbox("Time Series (ARIMA)", value=True)
        ts_params = {}
        if train_ts:
            st.subheader("Time Series Parameters")
            ts_params['seasonal_period'] = st.slider("Seasonal Period (weeks)", 10, 30, 20)
            ts_params['diff_order'] = st.slider("Differencing Order", 0, 2, 1)
    
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            # Initialize model instances
            if train_rf:
                st.session_state.rf_model = RandomForestModel(**rf_params)
                
                # Train model and save train/test data
                st.session_state.X_train, st.session_state.X_test, \
                st.session_state.y_train, st.session_state.y_test, \
                st.session_state.test_data = st.session_state.rf_model.prepare_data(st.session_state.df)
                
                st.session_state.rf_model.train(
                    st.session_state.X_train, 
                    st.session_state.y_train
                )
            
            if train_ts:
                st.session_state.ts_model = TimeSeriesModel(**ts_params)
                
                # Train time series model
                st.session_state.ts_model.prepare_data(st.session_state.df)
                st.session_state.ts_model.train()
            
            st.session_state.model_trained = True
        
        st.success("Models trained successfully!")
        
        # Show feature importance if RF model trained
        if train_rf:
            st.subheader("Random Forest Feature Importance")
            fig = st.session_state.rf_model.plot_feature_importance()
            st.pyplot(fig)

# Predictions & Evaluation Page
elif app_mode == "Predictions & Evaluation":
    st.header("Predictions & Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("Please train models first!")
        st.stop()
    
    if st.button("Generate Predictions & Evaluate"):
        with st.spinner("Generating predictions..."):
            results = []
            
            # Random Forest predictions
            if hasattr(st.session_state, 'rf_model'):
                rf_preds = st.session_state.rf_model.predict(st.session_state.X_test)
                rf_wmae = evaluate_model(
                    st.session_state.y_test, 
                    rf_preds, 
                    st.session_state.test_data['IsHoliday'] if 'IsHoliday' in st.session_state.test_data.columns else None
                )
                results.append(("Random Forest", rf_wmae))
                
                # Plot predictions
                st.subheader("Random Forest Predictions")
                fig = plt.figure(figsize=(12, 6))
                plt.plot(st.session_state.y_test.values[:50], label='Actual')
                plt.plot(rf_preds[:50], label='Predicted')
                plt.legend()
                plt.title('Random Forest: Actual vs Predicted Sales (First 50 samples)')
                plt.xlabel('Sample Index')
                plt.ylabel('Weekly Sales')
                plt.grid(True)
                st.pyplot(fig)
            
            # Time Series predictions
            if hasattr(st.session_state, 'ts_model'):
                ts_preds, ts_wmae = st.session_state.ts_model.predict_and_evaluate()
                results.append(("Time Series (ARIMA)", ts_wmae))
                
                # Plot predictions
                st.subheader("Time Series Predictions")
                fig = st.session_state.ts_model.plot_predictions()
                st.pyplot(fig)
            
            # Display results table
            st.subheader("Model Comparison")
            results_df = pd.DataFrame(results, columns=["Model", "WMAE"])
            st.dataframe(results_df)
            
            # Highlight best model
            best_model = min(results, key=lambda x: x[1])
            st.success(f"Best model: {best_model[0]} with WMAE: {best_model[1]:.2f}")
            
            st.session_state.predictions_made = True
            
        # Offer download of predictions
        if st.session_state.predictions_made:
            st.subheader("Download Predictions")
            
            # Create downloadable CSV
            if hasattr(st.session_state, 'rf_model'):
                pred_df = pd.DataFrame({
                    'Actual': st.session_state.y_test.values,
                    'RF_Predicted': st.session_state.rf_model.predict(st.session_state.X_test)
                })
                
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"walmart_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
**Walmart Sales Forecasting App** 
""")
