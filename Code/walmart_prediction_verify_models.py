#!/usr/bin/env python
"""
Verification script to check if models are properly placed and loadable
Run this before deployment to ensure everything is set up correctly
"""

import os
import sys
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def check_file_exists(filepath):
    """Check if file exists and return its size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return True, size
    return False, 0

def verify_model(filepath, model_type):
    """Verify that a model can be loaded successfully"""
    try:
        # Load model using joblib for both types
        model = joblib.load(filepath)
        print(f"   ✅ {model_type} model loaded successfully")
        print(f"      Type: {type(model)}")
        
        if model_type == "auto_arima" and hasattr(model, 'order'):
            print(f"      Order: {model.order}")
        elif model_type == "exponential_smoothing" and hasattr(model, 'params'):
            print(f"      Parameters: {model.params}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error loading {model_type} model: {str(e)}")
        return False

def main():
    print("🔍 Verifying models for Walmart Sales Prediction App")
    print("=" * 50)
    
    # Define models to check
    models = [
        ("models/default/auto_arima.pkl", "auto_arima"),
        ("models/default/exponential_smoothing.pkl", "exponential_smoothing")
    ]
    
    all_passed = True
    
    for filepath, model_type in models:
        print(f"\nChecking {filepath}:")
        
        # Check if file exists
        exists, size = check_file_exists(filepath)
        if exists:
            print(f"   ✅ File exists (Size: {size/1024:.1f} KB)")
        else:
            print(f"   ❌ File not found!")
            all_passed = False
            continue
        
        # Verify model can be loaded
        if not verify_model(filepath, model_type):
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All verifications passed! Ready for deployment.")
    else:
        print("❌ Some verifications failed. Please fix the issues before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()