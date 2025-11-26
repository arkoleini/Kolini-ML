# Verification Approach
# For checking assumptions programmatically:

# Linearity - Use scatter plots or residual plots
# Independence of Errors - Durbin-Watson test
# Homoscedasticity - Residual plots, Breusch-Pagan test
# Normality of Errors - Q-Q plots, Shapiro-Wilk test, histograms

# # Here's a practical code example that checks all these assumptions:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class RegressionAssumptionChecker:
    """Check if data meets Linear Regression assumptions"""
                #     __init__(self, X, y)
                # Initializes the checker by:

                # Storing your input data (X = features, y = target)
                # Training a LinearRegression model immediately
                # Calculating predictions and residuals (errors = actual - predicted)
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.y_pred = self.model.predict(X)
        self.residuals = y - self.y_pred
        
                # check_all() Master function that runs all four assumption checks in sequence and displays diagnostic plots. It's your one-stop function to verify everything.
    def check_all(self):
        """Run all assumption checks"""
        print("=" * 60)
        print("LINEAR REGRESSION ASSUMPTION CHECKS")
        print("=" * 60)
        
        self.check_linearity()
        self.check_independence()
        self.check_homoscedasticity()
        self.check_normality()
        
        self.plot_diagnostics()
        
    def check_linearity(self):
        """Check for linear relationship"""
        print("\n1. LINEARITY CHECK")
        print("-" * 60)
        
        # Calculate correlation for each feature
        
        
        
                  # Tests Assumption #1: Linear Relationship  (R)
                  # For single feature: calculates correlation coefficient between X and y
                  # Correlation > 0.7 = strong linear relationship ✓
                  # Correlation 0.3-0.7 = moderate relationship ⚠
                  # Correlation < 0.3 = weak relationship, consider non-linear models ✗
                  
        if self.X.shape[1] == 1:
            corr = np.corrcoef(self.X.flatten(), self.y)[0, 1]
            print(f"   Correlation: {corr:.4f}")
            if abs(corr) > 0.7:
                print("   ✓ Strong linear relationship detected")
            elif abs(corr) > 0.3:
                print("   ⚠ Moderate linear relationship")
            else:
                print("   ✗ Weak linear relationship - consider non-linear model")
        else:
            print("   Multiple features detected - check residual plot")
            
            
            
            
                  # Tests Assumption #2: Independence of Errors
                  # Uses Durbin-Watson test (checks if errors are correlated with each other)
                  # DW statistic between 1.5-2.5 = errors are independent ✓
                  # DW outside this range = autocorrelation exists (errors depend on each other) ✗
                  # Important for time-series data or ordered observations            
    def check_independence(self):
        """Check independence of errors using Durbin-Watson test"""
        print("\n2. INDEPENDENCE OF ERRORS")
        print("-" * 60)
        
        dw = durbin_watson(self.residuals)
        print(f"   Durbin-Watson statistic: {dw:.4f}")
        
        if 1.5 < dw < 2.5:
            print("   ✓ No significant autocorrelation detected")
        else:
            print("   ✗ Possible autocorrelation - errors may not be independent")
            
            
            
                  # Tests Assumption #3: Constant Variance

                  # Uses Breusch-Pagan test to check if error variance is constant across all X values
                  # p-value > 0.05 = variance is constant (homoscedasticity) ✓
                  # p-value < 0.05 = variance changes (heteroscedasticity) ✗
                  # Matches your Image 3 showing variance should be consisten
    def check_homoscedasticity(self):
        """Check constant variance of errors"""
        print("\n3. HOMOSCEDASTICITY (Constant Variance)")
        print("-" * 60)
        
        # Breusch-Pagan test requires a constant column
        import statsmodels.api as sm
        X_with_const = sm.add_constant(self.X)
        
        # Breusch-Pagan test
        bp_test = het_breuschpagan(self.residuals, X_with_const)
        bp_stat, bp_pvalue = bp_test[0], bp_test[1]
        
        print(f"   Breusch-Pagan test p-value: {bp_pvalue:.4f}")
        
        if bp_pvalue > 0.05:
            print("   ✓ Homoscedasticity assumption satisfied")
        else:
            print("   ✗ Heteroscedasticity detected - variance not constant")
         
         
         
                  # Tests Assumption #4: Normally Distributed Errors
                  # Shapiro-Wilk test: statistical test for normality
                  # p-value > 0.05 = errors are normally distributed ✓
                  # p-value < 0.05 = errors not normal ✗
                  # Skewness: measures asymmetry (ideal = 0)
                  # Positive = right-skewed, Negative = left-skewed
                  # Kurtosis: measures "tailedness" (ideal = 0)
                  # Positive = heavy tails, Negative = light tails  
    def check_normality(self):
        """Check if errors are normally distributed"""
        print("\n4. NORMALITY OF ERRORS")
        print("-" * 60)
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_pvalue = stats.shapiro(self.residuals)
        
        print(f"   Shapiro-Wilk test p-value: {shapiro_pvalue:.4f}")
        
        if shapiro_pvalue > 0.05:
            print("   ✓ Errors appear normally distributed")
        else:
            print("   ⚠ Errors may not be normally distributed")
            print("   (Note: Large samples may show significant results)")
            
        # Skewness and Kurtosis
        skew = stats.skew(self.residuals)
        kurt = stats.kurtosis(self.residuals)
        
        print(f"   Skewness: {skew:.4f} (ideal: close to 0)")
        print(f"   Kurtosis: {kurt:.4f} (ideal: close to 0)")
       
       
       
       
       
###   ----Visualization Methods-----------
        
    def plot_diagnostics(self):
        """Create diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(self.y_pred, self.residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted\n(Check Linearity & Homoscedasticity)')
        
        # 2. Q-Q Plot
        stats.probplot(self.residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot\n(Check Normality)')
        
        # 3. Scale-Location Plot
        standardized_residuals = np.sqrt(np.abs(self.residuals / np.std(self.residuals)))
        axes[1, 0].scatter(self.y_pred, standardized_residuals, alpha=0.6)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location Plot\n(Check Homoscedasticity)')
        
        # 4. Histogram of Residuals
        axes[1, 1].hist(self.residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Histogram of Residuals\n(Check Normality)')
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking - don't wait for window to close
        plt.pause(0.1)  # Brief pause to ensure plot displays
        
    def summary(self):
        """Provide overall recommendation"""
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print("Review the checks above and diagnostic plots.")
        print("If multiple assumptions are violated, consider:")
        print("  • Data transformation (log, sqrt, Box-Cox)")
        print("  • Non-linear models (polynomial, decision trees)")
        print("  • Regularization (Ridge, Lasso)")
        print("  • Different model type entirely")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10
    y = 3 * X.flatten() + 7 + np.random.randn(100) * 5
    
    # Run checks
    checker = RegressionAssumptionChecker(X, y)
    checker.check_all()
    checker.summary()