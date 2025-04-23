import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

class RandomForestModel:
    """Random Forest model for Walmart sales prediction"""
    
    def __init__(self, n_estimators=50, max_depth=35, max_features='sqrt', random_state=42):
        """Initialize the Random Forest model"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
            min_samples_split=10
        )
        self.scaler = RobustScaler()
        self.pipeline = make_pipeline(self.scaler, self.rf)
        
    def prepare_data(self, df, target_col="Weekly_Sales"):
        """Prepare data for training and testing"""
        df_encoded = df.copy()
        
        # Encode categorical variables if they exist
        if 'Type' in df_encoded.columns and df_encoded['Type'].dtype == 'object':
            type_group = {'A': 1, 'B': 2, 'C': 3}
            df_encoded['Type'] = df_encoded['Type'].replace(type_group).astype(int)
        
        # Convert boolean columns to integers if they exist
        for col in ['Super_Bowl', 'Thanksgiving', 'Labor_Day', 'Christmas', 'IsHoliday']:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(bool).astype(int)
        
        # Sort by date
        if 'Date' in df_encoded.columns:
            df_encoded = df_encoded.sort_values(by='Date', ascending=True)
        
        # Train-test split (70-30)
        train_data = df_encoded[:int(0.7 * len(df_encoded))]
        test_data = df_encoded[int(0.7 * len(df_encoded)):]
        
        # Define features and target
        used_cols = [c for c in df_encoded.columns if c != target_col]
        X_train = train_data[used_cols]
        X_test = test_data[used_cols]
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        
        # Drop Date column if it exists
        if 'Date' in X_train.columns:
            X_train = X_train.drop(['Date'], axis=1)
            X_test = X_test.drop(['Date'], axis=1)
        
        self.feature_names = X_train.columns
        return X_train, X_test, y_train, y_test, test_data
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        self.pipeline.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions with the trained model"""
        return self.pipeline.predict(X_test)
    
    def plot_feature_importance(self):
        """Plot feature importance from the trained model"""
        # Extract the RF model from the pipeline
        rf = self.pipeline.named_steps['randomforestregressor']
        
        # Get feature importances
        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Feature Importances")
        ax.bar(range(len(self.feature_names)), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(self.feature_names)), [self.feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(self.feature_names)])
        plt.tight_layout()
        return fig


class TimeSeriesModel:
    """Time Series (ARIMA) model for Walmart sales prediction"""
    
    def __init__(self, seasonal_period=20, diff_order=1):
        """Initialize the Time Series model"""
        self.seasonal_period = seasonal_period
        self.diff_order = diff_order
        self.model = None
        
    def prepare_data(self, df):
        """Prepare data for time series analysis"""
        # Convert Date to datetime if it's not already
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df["Date"] = pd.to_datetime(df["Date"])
        
        # Make a copy to avoid modifying the original
        df_ts = df.copy()
        
        # Set Date as index if it's not already
        if 'Date' in df_ts.columns:
            df_ts.set_index('Date', inplace=True)
        
        # Create weekly aggregated data
        self.df_week = df_ts.select_dtypes(include='number').resample('W').mean()
        
        # Apply differencing
        self.df_week_diff = self.df_week['Weekly_Sales'].diff().dropna()
        
        # Train/test split
        self.train_size = int(0.7 * len(self.df_week_diff))
        self.train_data_diff = self.df_week_diff[:self.train_size]
        self.test_data_diff = self.df_week_diff[self.train_size:]
        self.test_data_original = self.df_week['Weekly_Sales'][self.train_size:]
        
        return self.train_data_diff, self.test_data_diff
    
    def train(self):
        """Train the time series model using auto_arima"""
        # Use auto_arima to find best parameters
        self.model = auto_arima(
            self.train_data_diff, 
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=True,
            m=self.seasonal_period,
            d=self.diff_order,
            D=1,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        # Fit the model
        self.model.fit(self.train_data_diff)
        return self
    
    def predict_and_evaluate(self):
        """Make predictions and evaluate the model"""
        # Generate predictions
        predictions = self.model.predict(n_periods=len(self.test_data_diff))
        
        # Calculate WMAE
        wmae = self._calculate_wmae(self.test_data_diff.values, predictions)
        
        self.predictions = predictions
        return predictions, wmae
    
    def _calculate_wmae(self, y_true, y_pred):
        """Calculate weighted mean absolute error"""
        weights = np.ones_like(y_true)
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    
    def plot_predictions(self):
        """Plot the time series predictions vs actual values"""
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.set_title('Prediction of Weekly Sales Using Time Series Model', fontsize=20)
        ax.plot(self.train_data_diff, label='Train')
        ax.plot(self.test_data_diff, label='Test')
        
        # Convert predictions to pandas Series with matching index
        y_pred_series = pd.Series(self.predictions, index=self.test_data_diff.index)
        ax.plot(y_pred_series, label='Prediction')
        
        ax.legend(loc='best')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Differenced Weekly Sales', fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        return fig
