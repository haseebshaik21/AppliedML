import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Read datasets
train = pd.read_csv('train.csv')
train_data = pd.read_csv('my_train.csv')
dev_data = pd.read_csv('my_dev.csv')
test_data = pd.read_csv('test.csv')

# Convert all fields to strings
train, train_data, dev_data, test_data = train.astype(str), train_data.astype(str), dev_data.astype(str), test_data.astype(str)

# Separate input and output columns
X, X_train, X_dev, X_test = train.drop(columns=['Id', 'SalePrice']), train_data.drop(columns=['Id', 'SalePrice']), dev_data.drop(columns=['Id', 'SalePrice']), test_data.drop(columns=['Id'])

# Identify categorical columns
cols = X_train.select_dtypes(include=['object']).columns

# Create and fit OneHotEncoder
cols_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
cols_transformer.fit(X_train)

# Transform the data
X_binarized = cols_transformer.transform(X)
X_train_binarized = cols_transformer.transform(X_train)
X_dev_binarized = cols_transformer.transform(X_dev)
X_test_binarized = cols_transformer.transform(X_test)

# Logarithmic transformation of target variables
y_train_log = np.log1p(pd.to_numeric(train_data['SalePrice']))
y_dev_log = np.log1p(pd.to_numeric(dev_data['SalePrice']))

# Create and fit Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_binarized, y_train_log)

# Predictions using Linear Regression
train_predictions_log = linear_reg_model.predict(X_train_binarized)
dev_predictions_log = linear_reg_model.predict(X_dev_binarized)

# Convert predictions back to the original scale
train_predictions = np.expm1(train_predictions_log)
dev_predictions = np.expm1(dev_predictions_log)

# Evaluate Linear Regression model
train_rmse_log = np.sqrt(mean_squared_error(y_train_log, train_predictions_log))
dev_rmse_log = np.sqrt(mean_squared_error(y_dev_log, dev_predictions_log))

# Ridge Regression with hyperparameter tuning
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'alpha': alphas}

ridge_model = Ridge()
grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train_binarized, y_train_log)

best_alpha = grid_search.best_params_['alpha']
best_ridge_model = Ridge(alpha=best_alpha)
best_ridge_model.fit(X_train_binarized, y_train_log)

# Predictions using Ridge Regression
train_predictions_log_ridge = best_ridge_model.predict(X_train_binarized)
dev_predictions_log_ridge = best_ridge_model.predict(X_dev_binarized)

# Convert Ridge predictions back to original scale
train_predictions_ridge = np.expm1(train_predictions_log_ridge)
dev_predictions_ridge = np.expm1(dev_predictions_log_ridge)

# Evaluate Ridge Regression
train_rmse_log_ridge = np.sqrt(mean_squared_error(y_train_log, train_predictions_log_ridge))
dev_rmse_log_ridge = np.sqrt(mean_squared_error(y_dev_log, dev_predictions_log_ridge))

# Predictions on test data using Ridge model
test_predictions_log_ridge = best_ridge_model.predict(X_test_binarized)
test_predictions_ridge = np.expm1(test_predictions_log_ridge)

# Create a DataFrame with Id and SalePrice columns
ridge_predictions_df = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions_ridge
})

# Save predictions to a CSV file
ridge_predictions_df.to_csv('ridge_predictions.csv', index=False)
