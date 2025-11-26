import numpy as np
import pandas as pd
import importlib

# Import from files with hyphens using importlib
validate_scikit = importlib.import_module('Validate-DataSet-Scikit')
validate_statsmodels = importlib.import_module('Validate-DataSet-Statesmodels')

quick_regression_check = validate_scikit.quick_regression_check
RegressionAssumptionChecker = validate_statsmodels.RegressionAssumptionChecker


def validate_linear_regression(X, y):
    """
    Complete validation workflow combining both approaches
    """
    print("=" * 70)
    print("STEP 1: CHECK STATISTICAL ASSUMPTIONS")
    print("=" * 70)
    
    # Check if Linear Regression is appropriate
    checker = RegressionAssumptionChecker(X, y)
    checker.check_all()
    
    # Manual decision point
    proceed = input("\nDo assumptions look satisfied? (yes/no): ")
    
    if proceed.lower() != 'yes':
        print("\n⚠ WARNING: Proceeding with violated assumptions may lead to:")
        print("  - Biased predictions")
        print("  - Invalid confidence intervals")
        print("  - Poor generalization")
        print("\nConsider: transformation, non-linear models, or regularization")
        return
    
    print("\n" + "=" * 70)
    print("STEP 2: EVALUATE PREDICTIVE PERFORMANCE")
    print("=" * 70)
    
    # Use the quick_regression_check function from Validate_DataSet_Scikit
    mean_r2 = quick_regression_check(X, y)
    
    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)
    if mean_r2 > 0.7:
        print("\n✓ CONCLUSION: Model is statistically valid AND predictive")
    elif mean_r2 > 0.4:
        print("\n⚠ CONCLUSION: Model is valid but moderate predictive power")
        print("  Consider: adding features, feature engineering")
    else:
        print("\n✗ CONCLUSION: Model is valid but poor predictive power")
        print("  Consider: different features, more data, or different model")

# Load diabetes dataset
if __name__ == "__main__":
    # Load dataset from URL
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    
    # Display dataset info
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Prepare features (X) and target (y)
    # Predicting Glucose levels (continuous variable - suitable for linear regression)
    # Using BMI, Age, Pregnancies, Insulin, BloodPressure as predictors
    X = df[['BMI', 'Age', 'Pregnancies', 'Insulin', 'BloodPressure']].values
    y = df['Glucose'].values
    
    print(f"\nTarget variable: Glucose")
    print(f"Features: BMI, Age, Pregnancies, Insulin, BloodPressure")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\n" + "=" * 70)
    
    # Run validation
    validate_linear_regression(X, y)