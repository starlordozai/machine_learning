import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Generate Simulated Data (Replace this with your actual data collection)
np.random.seed(42) # For reproducibility
n_samples = 500

data = pd.DataFrame({
    'Popliteal_Height': np.random.normal(42, 3, n_samples),  # Mean 42cm, SD 3cm
    'Buttock_Popliteal_Length': np.random.normal(45, 4, n_samples),
    'Elbow_Rest_Height': np.random.normal(22, 2, n_samples),
    'Seat_Height': np.random.normal(43, 5, n_samples), # Furniture is one-size-fits-all
    'Seat_Depth': np.random.normal(38, 5, n_samples),
    'Desk_Height': np.random.normal(70, 5, n_samples),
})

# 2. Calculate Mismatch Ratios/Differences (Feature Engineering)
data['SH_PH_Ratio'] = data['Seat_Height'] / data['Popliteal_Height']
data['SD_BPL_Ratio'] = data['Seat_Depth'] / data['Buttock_Popliteal_Length']
data['DH_ERH_Diff'] = data['Desk_Height'] - data['Elbow_Rest_Height']

# 3. Define the Binary Target Variable 'Suitable' (Y)
# Based on our earlier definition
data['Suitable_SH'] = (data['Seat_Height'] >= 0.9 * data['Popliteal_Height']) & (data['Seat_Height'] <= 1.1 * data['Popliteal_Height'])
data['Suitable_SD'] = (data['Seat_Depth'] <= 0.8 * data['Buttock_Popliteal_Length'])
data['Suitable_DH'] = (data['Desk_Height'] >= data['Elbow_Rest_Height'] - 2) & (data['Desk_Height'] <= data['Elbow_Rest_Height'] + 3)

data['Suitable'] = (data['Suitable_SH'] & data['Suitable_SD'] & data['Suitable_DH']).astype(int) # 1 if all True, else 0

# 4. Prepare Data for Modeling
# Select our engineered features and the target
X = data[['SH_PH_Ratio', 'SD_BPL_Ratio', 'DH_ERH_Diff']]
y = data['Suitable']

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# 5. Split the Data (Optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build and Train the Logit Model
model = sm.Logit(y_train, X_train)
result = model.fit()

# 7. View Model Results
print(result.summary2())

# 8. Interpret the Results
# Look at the p-values (P>|z|) to see which features are significant predictors.
# A coefficient (Coef.) for 'SH_PH_Ratio' of -5.0 would mean:
# "For every one-unit increase in the Seat-Height-to-Popliteal-Height ratio,
# the log-odds of the furniture being suitable decrease by 5."
# Since the ideal ratio is ~1, we'd expect values far from 1 to have negative coefficients.

# 9. Make Predictions on the Test Set (Evaluation)
# Get predicted probabilities
y_pred_proba = result.predict(X_test)
# Convert probabilities to class predictions (using 0.5 as threshold)
y_pred = (y_pred_proba >= 0.5).astype(int)

# 10. Evaluate Model Performance
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))