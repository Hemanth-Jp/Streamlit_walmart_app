# Deployment Guide for Walmart Sales Prediction App

## Streamlit Community Cloud Deployment

### Step 1: Prepare Your Repository

1. **Create a GitHub repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/walmart-sales-prediction.git
   git push -u origin main
   ```

2. **Ensure these files are included:**
   ```
   streamlit_app_prediction/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   └── models/default/
       ├── auto_arima.pkl
       └── exponential_smoothing.pkl
   ```

### Step 2: Deploy to Streamlit Community Cloud

1. **Sign up/Log in**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create new app**
   - Click "New app" button
   - Select repository: `your-username/walmart-sales-prediction`
   - Select branch: `main`
   - Set main file path: `streamlit_app_prediction/app.py`
   - App name: `walmart-sales-prediction` (or your choice)

3. **Configure environment**
   - Streamlit will automatically detect `requirements.txt`
   - No additional configuration needed

4. **Deploy**
   - Click "Deploy!" button
   - Wait for deployment to complete (usually 2-3 minutes)
   - App will be available at: `https://walmart-sales-prediction.streamlit.app`

### Step 3: Post-Deployment Setup

1. **Verify default models**
   - Ensure models/default/ directory contains the .pkl files
   - Test default model loading functionality

2. **Test all features**
   - Test default model loading
   - Test file upload functionality
   - Generate predictions
   - Download results

### Important Deployment Notes

1. **File Size Limitations**
   - Streamlit Community Cloud has a 1GB limit per app
   - Large model files may need compression

2. **Default Models**
   - Must be included in the repository
   - Will be deployed with the app
   - Can't be updated without redeployment

3. **Session State**
   - Uploaded models are session-specific
   - Sessions reset after inactivity
   - Default models persist between sessions

### Troubleshooting Deployment

1. **Deployment Failed**
   - Check requirements.txt format
   - Verify all dependencies are specified
   - Ensure no syntax errors in app.py

2. **Models Not Found**
   - Verify models/default/ directory exists
   - Check file permissions
   - Ensure .pkl files are included in git

3. **Import Errors**
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt
   - Test locally before deployment

### Updating the Deployed App

1. **Make changes locally**
   ```bash
   # Edit files
   git add .
   git commit -m "Update description"
   git push origin main
   ```

2. **Automatic deployment**
   - Streamlit detects changes automatically
   - Rebuilds and redeploys within minutes
   - No manual intervention needed

### Managing Secrets

For sensitive data (if needed in future):

1. **Add secrets file**
   ```toml
   # .streamlit/secrets.toml
   API_KEY = "your-api-key"
   ```

2. **Access in code**
   ```python
   api_key = st.secrets["API_KEY"]
   ```

### Performance Optimization

1. **Model Loading**
   - Models are loaded once per session
   - Use session state to prevent reloading
   - Consider model compression

2. **Caching**
   ```python
   @st.cache_data
   def load_default_model(model_type):
       # Cached loading logic
   ```

### Monitoring

1. **View Logs**
   - Access logs from Streamlit dashboard
   - Check for runtime errors
   - Monitor performance metrics

2. **Usage Analytics**
   - Track user interactions
   - Monitor load times
   - Review error rates

### Support Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Community Forums](https://discuss.streamlit.io)
- [Deployment FAQ](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

## Next Steps

After successful deployment:
1. Share the app URL with stakeholders
2. Monitor initial user feedback
3. Plan future enhancements
4. Set up regular model updates

Your app will be live and accessible to anyone with the URL!