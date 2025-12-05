"""
Train AQI prediction model from the dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data(data_path='data/city_day.csv'):
    """Load and preprocess the air quality data"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Rows with AQI: {df['AQI'].notna().sum()}")
    
    df = df[df['AQI'].notna()].copy()
    
    pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                         'Benzene', 'Toluene', 'Xylene']
    
    X = df[pollutant_features].copy()
    y = df['AQI'].copy()
    
    print("\nHandling missing values...")
    median_values = {}
    for col in pollutant_features:
        missing_count = X[col].isna().sum()
        median_val = X[col].median()
        median_values[col] = median_val
        if missing_count > 0:
            X[col] = X[col].fillna(median_val)
            print(f"  {col}: Filled {missing_count} missing values with median {median_val:.2f}")
    
    valid_mask = ~X.isna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Target (AQI) statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    
    return X, y, pollutant_features, median_values

def train_model(X, y):
    """Train Random Forest model for AQI prediction"""
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  MAE: {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²: {test_r2:.4f}")
    
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model

def save_model(model, pollutant_features, median_values, model_path='models/aqi_model.pkl'):
    """Save the trained model, feature list, and median values"""
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'model': model,
        'features': pollutant_features,
        'median_values': median_values
    }, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    print("=" * 60)
    print("AQI Prediction Model Training")
    print("=" * 60)
    
    X, y, pollutant_features, median_values = load_and_preprocess_data()
    model = train_model(X, y)
    save_model(model, pollutant_features, median_values)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()

