from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import StandardScaler


housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

# Observations:
    # Dataset has 20,640 rows and 9 columns (8 features + 1 target).
    # All features are numeric (float).
    # There are no missing values.
    # Median income (MedInc) appears to vary significantly.
    # Longitude and Latitude may strongly influence house value due to location.

print("Nulls before:", df.isnull().sum().sum())

df = df.dropna()

print("Nulls after:", df.isnull().sum().sum())

# I used dropna() because the dataset contains no missing values.
# In real financial datasets, I might prefer fillna() to preserve data.

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Hypothesis:
    # I believe MedInc, Latitude, and Longitude will most influence house value
    # because income and location typically drive real estate pricing.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score : {r2:.4f}")
print(f"MAE : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

print("\nFeature Importance:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

sample = X_test.iloc[:3]
predictions = model.predict(sample)
actuals = y_test.iloc[:3].values

for i in range(3):
    print(f"Prediction: {predictions[i]:.2f} | Actual: {actuals[i]:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual House Value ($100k)")
plt.ylabel("Predicted House Value ($100k)")
plt.title("Linear Regression — Actual vs Predicted")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()


# Model Interpretation:
    # The MAE of 0.5332 indicates the model's predictions are off by
    # approximately $53,320 on average.
    # The RMSE of 0.7456 suggests some larger prediction errors exist.
    # Overall performance falls within the expected range for this dataset,
    # indicating the linear regression model captures general trends
    # but does not perfectly predict extreme housing values.


# Model Performance Interpretation:
    # The R² score of 0.5758 indicates the model explains approximately 58%
    # of the variation in housing prices. While slightly below the expected
    # range, this is typical for linear regression on real-world housing data,
    # where many external factors influence price beyond the available features.
    # The MAE and RMSE values fall within expected ranges, suggesting the model
    # makes reasonably accurate predictions overall.


#Feature selection: Drop 3 features and retrain. Did your R2 go up or down? Why?
print("\n--- Feature Selection Experiment ---")
X_reduced = df.drop(columns=[
    'MedHouseVal',
    'Population',
    'AveOccup',
    'HouseAge'
])
y = df['MedHouseVal']

#Split Again
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

#Train again
model2 = LinearRegression()
model2.fit(X_train2, y_train2)

#Evaluate again
y_pred2 = model2.predict(X_test2)
r2_new = r2_score(y_test2, y_pred2)

print("New R2 after dropping features:", r2_new)

# Feature Selection Interpretation:
    # After removing three features and retraining the model, the R² score
    # decreased from 0.5758 to 0.5675. This indicates the removed features
    # contributed useful information to predicting house values.
    # Even variables that appear less important can improve overall model
    # performance when combined with other features.


#Scaling: Add StandardScaler before training. Does scaling change linear regression results?

print("\n--- Scaling Experiment ---")

#This creates a StandardScaler instance to standardize the features before training the model.
scaler = StandardScaler()

# This line applies the scaler to the feature matrix X, transforming the data to have a mean of 0 and a standard deviation of 1. NOT Y
X_scaled = scaler.fit_transform(X)

#Split Again
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#Train again
model3 = LinearRegression()
model3.fit(X_train3, y_train3)

#Evaluate again
y_pred3 = model3.predict(X_test3)
r2_scaled = r2_score(y_test3, y_pred3)
print("R2 after scaling:", r2_scaled)

# Scaling Interpretation:
    # After applying StandardScaler and retraining the model, the R² score
    # remained nearly the same (0.5758). This shows that scaling does not
    # significantly impact linear regression performance because the model
    # relies on linear relationships rather than feature magnitude.
    # Scaling mainly affects coefficient size rather than predictive accuracy.


#Residuals: Plot the distribution of (y_test - y_pred). What does the shape tell you?

# This calculates the residuals, which are the differences between the actual values (y_test) and the predicted values (y_pred). Residuals help us understand how well the model is performing. If the residuals are normally distributed around zero, it suggests that the model is capturing the underlying patterns in the data well. However, if there are patterns in the residuals (e.g., skewness, heteroscedasticity), it may indicate that the model is not fully capturing the relationships in the data or that there are outliers affecting performance.
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=40)
plt.title("Distribution of Residuals")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt
plt.savefig("residual_distribution.png", dpi=150)

# Residual Interpretation:
    # The residual plot shows that most prediction errors are centered around zero,
    # meaning the model's predictions are generally close to actual values.
    # The distribution is fairly balanced, indicating the model does not consistently
    # overpredict or underpredict housing prices. Some larger errors exist,
    # which is expected in real-world housing data where prices are influenced by
    # many external factors not captured by the model.