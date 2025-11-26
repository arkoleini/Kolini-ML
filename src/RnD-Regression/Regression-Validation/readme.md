# Code Explanation - Regression Assumption Checker

## Main Class: `RegressionAssumptionChecker`

### **`__init__(self, X, y)`**

Initializes the checker by:

- Storing your input data (X = features, y = target)
- Training a LinearRegression model immediately
- Calculating predictions and residuals (errors = actual - predicted)

### **`check_all()`**

Master function that runs all four assumption checks in sequence and displays diagnostic plots. It's your one-stop function to verify everything.

---

## Individual Check Methods

### **`check_linearity()`**

**Tests Assumption #1: Linear Relationship**

- For single feature: calculates correlation coefficient between X and y
- Correlation > 0.7 = strong linear relationship ✓
- Correlation 0.3-0.7 = moderate relationship ⚠
- Correlation < 0.3 = weak relationship, consider non-linear models ✗

### **`check_independence()`**

**Tests Assumption #2: Independence of Errors**

- Uses Durbin-Watson test (checks if errors are correlated with each other)
- DW statistic between 1.5-2.5 = errors are independent ✓
- DW outside this range = autocorrelation exists (errors depend on each other) ✗
- Important for time-series data or ordered observations

### **`check_homoscedasticity()`**

**Tests Assumption #3: Constant Variance**

- Uses Breusch-Pagan test to check if error variance is constant across all X values
- p-value > 0.05 = variance is constant (homoscedasticity) ✓
- p-value < 0.05 = variance changes (heteroscedasticity) ✗
- Matches your Image 3 showing variance should be consistent

### **`check_normality()`**

**Tests Assumption #4: Normally Distributed Errors**

- **Shapiro-Wilk test**: statistical test for normality
  - p-value > 0.05 = errors are normally distributed ✓
  - p-value < 0.05 = errors not normal ✗
- **Skewness**: measures asymmetry (ideal = 0)
  - Positive = right-skewed, Negative = left-skewed
- **Kurtosis**: measures "tailedness" (ideal = 0)
  - Positive = heavy tails, Negative = light tails

---

## Visualization Method

### **`plot_diagnostics()`**

Creates 4 diagnostic plots to visually verify assumptions:

1. **Residuals vs Fitted Plot**

   - Check linearity: points should scatter randomly around horizontal line at 0
   - Check homoscedasticity: spread should be consistent across x-axis

2. **Q-Q Plot (Quantile-Quantile)**

   - Check normality: points should follow diagonal line
   - Deviations indicate non-normal distribution

3. **Scale-Location Plot**

   - Another check for homoscedasticity
   - Should see random scatter without patterns or trends

4. **Histogram of Residuals**
   - Visual check for normality
   - Should look like a bell curve (normal distribution)

### **`summary()`**

Provides recommendations if assumptions are violated, suggesting data transformations or alternative models.

---

## Helper Function: `quick_regression_check(X, y)`

A simpler alternative that:

- Performs 5-fold cross-validation
- Calculates R² scores (how well model explains variance)
- Gives quick verdict on model suitability:
  - R² > 0.7 = good fit ✓
  - R² 0.4-0.7 = moderate fit ⚠
  - R² < 0.4 = poor fit ✗

---

## Key Statistical Tests Used

| Test              | Purpose                               | Good Result    |
| ----------------- | ------------------------------------- | -------------- |
| **Correlation**   | Measures linear relationship strength | \|r\| > 0.7    |
| **Durbin-Watson** | Detects autocorrelation in residuals  | 1.5 < DW < 2.5 |
| **Breusch-Pagan** | Tests for heteroscedasticity          | p > 0.05       |
| **Shapiro-Wilk**  | Tests for normality                   | p > 0.05       |

---

## Summary

These functions together verify all four assumptions shown in your images before you commit to using Linear Regression:

1. ✓ **Linear Relationship** - Data follows a straight line pattern
2. ✓ **Independence of Errors** - Residuals are not correlated
3. ✓ **Homoscedasticity** - Constant variance across all values
4. ✓ **Normality of Errors** - Residuals follow normal distribution

If assumptions are violated, consider:

- Data transformations (log, sqrt, Box-Cox)
- Non-linear models (polynomial regression, decision trees)
- Regularization techniques (Ridge, Lasso)
- Alternative model types entirely
