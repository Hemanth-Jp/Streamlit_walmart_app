import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def preprocess_data(df_store, df_train, df_features):
    """
    Preprocess and clean the three input dataframes.
    
    Parameters:
    df_store -- Store information dataframe
    df_train -- Training sales dataframe
    df_features -- Additional features dataframe
    
    Returns:
    df -- Cleaned and merged dataframe
    """
    # Merge the three datasets
    df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
    
    # Handle duplicate columns after merge
    if 'IsHoliday_x' in df.columns and 'IsHoliday_y' in df.columns:
        df.drop(['IsHoliday_y'], axis=1, inplace=True)
        df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)
    
    # Remove rows with non-positive sales
    df = df.loc[df['Weekly_Sales'] > 0]
    
    # Fill missing values
    df = df.fillna(0)
    
    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Create time-based features
    df['week'] = df['Date'].dt.isocalendar().week
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    
    # Create specific holiday indicator columns
    # Super bowl dates
    df.loc[(df['Date'] == '2010-02-12') | (df['Date'] == '2011-02-11') | 
           (df['Date'] == '2012-02-10'), 'Super_Bowl'] = True
    df.loc[(df['Date'] != '2010-02-12') & (df['Date'] != '2011-02-11') & 
           (df['Date'] != '2012-02-10'), 'Super_Bowl'] = False
    
    # Labor day dates
    df.loc[(df['Date'] == '2010-09-10') | (df['Date'] == '2011-09-09') | 
           (df['Date'] == '2012-09-07'), 'Labor_Day'] = True
    df.loc[(df['Date'] != '2010-09-10') & (df['Date'] != '2011-09-09') & 
           (df['Date'] != '2012-09-07'), 'Labor_Day'] = False
    
    # Thanksgiving dates
    df.loc[(df['Date'] == '2010-11-26') | (df['Date'] == '2011-11-25'), 'Thanksgiving'] = True
    df.loc[(df['Date'] != '2010-11-26') & (df['Date'] != '2011-11-25'), 'Thanksgiving'] = False
    
    # Christmas dates
    df.loc[(df['Date'] == '2010-12-31') | (df['Date'] == '2011-12-30'), 'Christmas'] = True
    df.loc[(df['Date'] != '2010-12-31') & (df['Date'] != '2011-12-30'), 'Christmas'] = False
    
    return df

def create_visualizations(df, viz_type):
    """
    Create visualizations based on the visualization type.
    
    Parameters:
    df -- Input dataframe
    viz_type -- Type of visualization to create: 'store_analysis', 'holiday_analysis', 
                'time_analysis', or 'external_factors'
    
    Returns:
    Matplotlib figure(s)
    """
    if viz_type == "store_analysis":
        # Store type distribution
        fig1 = plt.figure(figsize=(10, 8))
        my_data = [df[df['Type'] == 'A'].shape[0] / df.shape[0] * 100,
                  df[df['Type'] == 'B'].shape[0] / df.shape[0] * 100,
                  df[df['Type'] == 'C'].shape[0] / df.shape[0] * 100]
        my_labels = 'Type A', 'Type B', 'Type C'
        plt.pie(my_data, labels=my_labels, autopct='%1.1f%%', 
                textprops={'fontsize': 15}, colors=['#8fbc8f', '#ffb6c1', '#cd5c5c'])
        plt.title('Store Type Distribution', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        
        # Store size by type
        fig2 = plt.figure(figsize=(10, 8))
        sns.boxplot(x='Type', y='Size', data=df, showfliers=False)
        plt.title('Store Size by Type')
        plt.xlabel('Store Type')
        plt.ylabel('Size')
        
        # Weekly sales by department
        fig3 = plt.figure(figsize=(20, 6))
        sns.barplot(x='Dept', y='Weekly_Sales', data=df)
        plt.title('Average Weekly Sales by Department')
        plt.xlabel('Department')
        plt.ylabel('Average Weekly Sales')
        
        return fig1, fig2, fig3
    
    elif viz_type == "holiday_analysis":
        # Holiday impact analysis
        fig1 = plt.figure(figsize=(10, 6))
        sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df)
        plt.title('Average Sales by Holiday Status')
        plt.xlabel('Is Holiday')
        plt.ylabel('Average Weekly Sales')
        
        # Compare average sales for each holiday type
        fig2 = plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sns.barplot(x='Christmas', y='Weekly_Sales', data=df)
        plt.title('Christmas Impact')

        plt.subplot(2, 2, 2)
        sns.barplot(x='Thanksgiving', y='Weekly_Sales', data=df)
        plt.title('Thanksgiving Impact')

        plt.subplot(2, 2, 3)
        sns.barplot(x='Super_Bowl', y='Weekly_Sales', data=df)
        plt.title('Super Bowl Impact')

        plt.subplot(2, 2, 4)
        sns.barplot(x='Labor_Day', y='Weekly_Sales', data=df)
        plt.title('Labor Day Impact')
        plt.tight_layout()
        
        return fig1, fig2
    
    elif viz_type == "time_analysis":
        # Monthly sales analysis
        monthly_sales = pd.pivot_table(df, values="Weekly_Sales", columns="year", index="month")
        fig1 = plt.figure(figsize=(12, 6))
        monthly_sales.plot()
        plt.title('Monthly Sales Trends by Year')
        plt.xlabel('Month')
        plt.ylabel('Average Weekly Sales')
        
        # Weekly sales analysis
        weekly_sales = pd.pivot_table(df, values="Weekly_Sales", columns="year", index="week")
        fig2 = plt.figure(figsize=(12, 6))
        weekly_sales.plot()
        plt.title('Weekly Sales Trends by Year')
        plt.xlabel('Week')
        plt.ylabel('Average Weekly Sales')
        
        return fig1, fig2
    
    elif viz_type == "external_factors":
        # External factors analysis
        fig = plt.figure(figsize=(16, 12))

        # Sales vs Fuel Price
        plt.subplot(2, 2, 1)
        fuel_price = pd.pivot_table(df, values="Weekly_Sales", index="Fuel_Price")
        plt.plot(fuel_price)
        plt.title('Sales vs Fuel Price')

        # Sales vs Temperature
        plt.subplot(2, 2, 2)
        temp = pd.pivot_table(df, values="Weekly_Sales", index="Temperature")
        plt.plot(temp)
        plt.title('Sales vs Temperature')

        # Sales vs CPI
        plt.subplot(2, 2, 3)
        CPI = pd.pivot_table(df, values="Weekly_Sales", index="CPI")
        plt.plot(CPI)
        plt.title('Sales vs CPI')

        # Sales vs Unemployment
        plt.subplot(2, 2, 4)
        unemployment = pd.pivot_table(df, values="Weekly_Sales", index="Unemployment")
        plt.plot(unemployment)
        plt.title('Sales vs Unemployment')

        plt.tight_layout()
        
        return fig
    
    return None

def evaluate_model(y_true, y_pred, holiday_col=None):
    """
    Calculate weighted mean absolute error (WMAE).
    
    Parameters:
    y_true -- True target values
    y_pred -- Predicted target values
    holiday_col -- Boolean column indicating holiday status (for weighting)
    
    Returns:
    wmae -- Weighted mean absolute error
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values
    
    # Set weights (5x for holidays, 1x for non-holidays)
    if holiday_col is None:
        weights = np.ones_like(y_true)
    else:
        if isinstance(holiday_col, (pd.Series, pd.DataFrame)):
            holiday_col = holiday_col.values
        weights = np.where(holiday_col, 5, 1)
    
    # Calculate WMAE
    error = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    return error

def generate_sample_data():
    """
    Generate sample data for demo purposes if no data is uploaded.
    
    Returns:
    df_store, df_train, df_features -- Sample dataframes
    """
    # Generate store data
    store_data = {
        'Store': np.arange(1, 11),
        'Type': np.random.choice(['A', 'B', 'C'], size=10),
        'Size': np.random.randint(30000, 200000, size=10)
    }
    df_store = pd.DataFrame(store_data)
    
    # Generate dates
    start_date = datetime(2010, 1, 1)
    dates = [start_date + pd.Timedelta(weeks=i) for i in range(100)]
    
    # Generate training data
    train_records = []
    for store in range(1, 11):
        for dept in range(1, 6):
            for date in dates[:50]:  # Use first 50 dates for training
                is_holiday = date.month in [2, 7, 11, 12] and date.day in [1, 15]
                weekly_sales = np.random.normal(20000, 5000) + \
                              (5000 if is_holiday else 0) + \
                              (store * 1000) + \
                              (dept * 500)
                train_records.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Weekly_Sales': max(0, weekly_sales),
                    'IsHoliday': is_holiday
                })
    df_train = pd.DataFrame(train_records)
    
    # Generate feature data
    feature_records = []
    for store in range(1, 11):
        for date in dates:
            feature_records.append({
                'Store': store,
                'Date': date.strftime('%Y-%m-%d'),
                'Temperature': np.random.uniform(30, 90),
                'Fuel_Price': np.random.uniform(2.5, 4.0),
                'MarkDown1': np.random.choice([0, np.random.uniform(1000, 5000)]),
                'MarkDown2': np.random.choice([0, np.random.uniform(1000, 5000)]),
                'MarkDown3': np.random.choice([0, np.random.uniform(1000, 5000)]),
                'MarkDown4': np.random.choice([0, np.random.uniform(1000, 5000)]),
                'MarkDown5': np.random.choice([0, np.random.uniform(1000, 5000)]),
                'CPI': np.random.uniform(200, 240),
                'Unemployment': np.random.uniform(5.0, 10.0),
                'IsHoliday': date.month in [2, 7, 11, 12] and date.day in [1, 15]
            })
    df_features = pd.DataFrame(feature_records)
    
    return df_store, df_train, df_features