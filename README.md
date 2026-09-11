# Predict New York City Airbnb Prices Using Linear Regression

## Overview
This project applies Linear Regression to predict New York City Airbnb listing prices based on available dataset features. It includes model training, performance evaluation using standard regression metrics, and visualization of prediction errors.

---

## Evaluation Metrics

* **R² Score:** `0.4708`
* **Mean Absolute Error (MAE):** `38.54`
* **Mean Squared Error (MSE):** `2834.96`
* **Root Mean Squared Error (RMSE):** `53.24`

### Interpretation
* **R² Score (0.4708):** Explains approximately 47% of the variance in Airbnb prices based on the dataset features.
* **MAE (38.54):** On average, predicted prices deviate from actual prices by about $38.54 USD.

---

## Code & Visualizations

### Task : Error Distribution Histogram
The code below calculates residual errors and plots a histogram showing the error distribution.

```python
import matplotlib.pyplot as plt

# Calculate residual errors
errors = y_test - y_pred

# Plot histogram of residual errors
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Error Distribution Histogram")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()


