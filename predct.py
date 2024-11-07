import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Load data
train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/project/train.csv")
test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/project/test.csv")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns in train set
train["Date"] = label_encoder.fit_transform(train["Date"])
train["Time"] = label_encoder.fit_transform(train["Time"])
train["State/UnionTerritory"] = label_encoder.fit_transform(train["State/UnionTerritory"])
train["ConfirmedIndianNational"] = label_encoder.fit_transform(train["ConfirmedIndianNational"])
train["ConfirmedForeignNational"] = label_encoder.fit_transform(train["ConfirmedForeignNational"])

train.info()

# Define features and target
features = train.drop(["Deaths"], axis=1)
target = train["Deaths"]

# Encode categorical columns in test set
test["Date"] = label_encoder.transform(test["Date"])
test["Time"] = label_encoder.transform(test["Time"])
test["State/UnionTerritory"] = label_encoder.transform(test["State/UnionTerritory"])
test["ConfirmedIndianNational"] = label_encoder.transform(test["ConfirmedIndianNational"])
test["ConfirmedForeignNational"] = label_encoder.transform(test["ConfirmedForeignNational"])

test.info()

# Split the data into training features and labels
X_train = features
y_train = target

# Define the model and parameter grid
rf_reg = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Initialize RandomForestRegressor with best parameters
best_rf_reg = RandomForestRegressor(**best_params, random_state=42)
best_rf_reg.fit(X_train, y_train)

# Predict on test data
predictions = best_rf_reg.predict(test)

# Add predictions to test set and save to CSV
test["Deaths"] = predictions
test.to_csv("project_predictions.csv", index=False)
