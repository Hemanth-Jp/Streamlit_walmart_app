# Walmart Sales Prediction App

This Streamlit application generates 4-week sales forecasts using trained time series models for Walmart sales data.

## Features

- Load pre-trained default models or upload custom models
- Generate 4-week sales forecasts
- Interactive visualization using Plotly
- Download predictions in CSV or JSON format
- View summary statistics for forecasts

## File Structure

```
streamlit_app_prediction/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/default/        # Default model storage directory
│   ├── auto_arima.pkl
│   └── exponential_smoothing.pkl
└── README.md             # This file
```

## Local Installation

1. Clone or download this repository
2. Navigate to the app directory:
   ```bash
   cd streamlit_app_prediction
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally

```bash
streamlit run app.py
```

Access the app at `http://localhost:8501`

## Deployment to Streamlit Community Cloud

### Prerequisites
- GitHub account
- Streamlit Community Cloud account

### Deployment Steps

1. **Push to GitHub**
   ```bash
   # Initialize repository
   git init
   git add .
   git commit -m "Initial commit"
   
   # Add remote and push
   git remote add origin https://github.com/your-username/your-repo.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set branch to `main`
   - Set main file path to `streamlit_app_prediction/app.py`
   - Click "Deploy!"

### Important Notes for Cloud Deployment
- The `models/default/` directory must contain pre-trained models
- Ensure all dependencies are listed in `requirements.txt`
- The app will be accessible at `https://<your-app-url>.streamlit.app`

## Usage Guide

### Loading Models

The app supports two methods for loading models:

1. **Default Models**
   - Pre-trained models saved in `models/default/`
   - Click "Load Auto ARIMA Model" or "Load Exponential Smoothing Model"

2. **Upload Custom Models**
   - Upload your own `.pkl` model files
   - Select model type (auto_arima or exponential_smoothing)
   - Click "Load Uploaded Model"

### Generating Predictions

1. Ensure a model is loaded (default or uploaded)
2. Click "Generate 4-Week Forecast"
3. View the interactive plot showing predicted sales
4. Download results in CSV or JSON format

### Understanding Results

- **Interactive Plot**: Shows predicted sales for next 4 weeks
- **Data Table**: Displays exact prediction values
- **Summary Statistics**: Shows total sales, average, highest week, and growth rate

## Model Compatibility

- **Auto ARIMA**: Must be saved using `joblib.dump()`
- **Exponential Smoothing**: Must be saved using model's `.save()` method

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| "Default model not found" | Missing default model file | Ensure models exist in `models/default/` |
| "Invalid model file" | Incorrect model format | Check model was saved correctly |
| "Error generating predictions" | Model incompatibility | Verify model type matches selection |

## Directory Structure Requirements

For cloud deployment, ensure:
```
streamlit_app_prediction/
├── app.py
├── requirements.txt
└── models/default/
    ├── auto_arima.pkl
    └── exponential_smoothing.pkl
```

## Development

### Testing Locally

1. Create sample models using the training app
2. Copy models to `models/default/`
3. Run the app and test both default and upload functionality

### Code Structure

- `load_default_model()`: Loads pre-saved models
- `load_uploaded_model()`: Handles user-uploaded models
- `predict_next_4_weeks()`: Generates forecasts
- `main()`: Streamlit UI and logic

## Troubleshooting

1. **Models not loading**
   - Check file paths
   - Verify model format
   - Ensure proper permissions

2. **Prediction errors**
   - Verify model compatibility
   - Check for missing dependencies
   - Ensure correct model type selected

3. **Deployment issues**
   - Verify all files are in repository
   - Check `requirements.txt` completeness
   - Ensure proper file paths for cloud

## Support

For issues:
1. Check error messages
2. Verify model formats
3. Review logs for detailed errors
4. Contact support if needed

## License

© 2025 Walmart Sales Forecasting Project