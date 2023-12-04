import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read datasets
train = pd.read_csv('train.csv')
train_data = pd.read_csv('my_train.csv')
dev_data = pd.read_csv('my_dev.csv')

# Convert all fields to strings
train, train_data, dev_data = train.astype(str), train_data.astype(str), dev_data.astype(str)

# Separate input and output columns
X, X_train, X_dev = train.drop(columns=['Id', 'SalePrice']), train_data.drop(columns=['Id', 'SalePrice']), dev_data.drop(columns=['Id', 'SalePrice'])

# Define categorical and numerical columns
categorical_cols = [...]
numerical_cols = [...]

# Create transformers
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
numerical_transformer = 'passthrough'
categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')

# ColumnTransformer with PolynomialFeatures and OneHotEncoder
all_fields_transformer = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_cols),
        ('categorical', categorical_transformer, categorical_cols),
        ('poly', poly_transformer, numerical_cols)  # Include the polynomial transformer
    ])

# Pipeline with the preprocessor and model
poly_reg_model = make_pipeline(poly_transformer, LinearRegression())
full_pipeline = Pipeline([
    ('preprocessor', all_fields_transformer),
    ('model', poly_reg_model)
])

# Fit the full pipeline on training data
full_pipeline.fit(X_train, np.log1p(pd.to_numeric(train_data['SalePrice'])))

# Transform the data
X_binarized = all_fields_transformer.transform(X)
X_train_binarized = all_fields_transformer.transform(X_train)
X_dev_binarized = all_fields_transformer.transform(X_dev)

# Predictions
train_predictions_log = full_pipeline.predict(X_train)
dev_predictions_log = full_pipeline.predict(X_dev)

# Convert predictions back to original scale
train_predictions = np.expm1(train_predictions_log)
dev_predictions = np.expm1(dev_predictions_log)

# Logarithmic transformation of target variables
y_train_log = np.log1p(pd.to_numeric(train_data['SalePrice']))
y_dev_log = np.log1p(pd.to_numeric(dev_data['SalePrice']))

# Evaluate model on log-transformed scale
train_rmse_log = np.sqrt(mean_squared_error(y_train_log, train_predictions_log))
dev_rmse_log = np.sqrt(mean_squared_error(y_dev_log, dev_predictions_log))

# Display results
print("\nTraining RMSE on log scale:", train_rmse_log)
print("Dev RMSE on log scale:", dev_rmse_log)
