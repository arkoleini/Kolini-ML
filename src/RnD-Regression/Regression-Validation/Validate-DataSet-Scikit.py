from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def quick_regression_check(X, y):
    """Quick check if Linear Regression is appropriate"""
    
    model = LinearRegression()
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean R²: {cv_scores.mean():.4f}")
    print(f"Std R²: {cv_scores.std():.4f}")
    
    if cv_scores.mean() > 0.7:
        print("✓ Model fits well - Linear Regression may be appropriate")
    elif cv_scores.mean() > 0.4:
        print("⚠ Moderate fit - investigate further")
    else:
        print("✗ Poor fit - consider alternative models")
    
    return cv_scores.mean()

# Usage example - only runs when this file is executed directly
if __name__ == "__main__":
    import numpy as np
    X = np.random.randn(100, 1)
    y = np.random.randn(100)
    quick_regression_check(X, y)