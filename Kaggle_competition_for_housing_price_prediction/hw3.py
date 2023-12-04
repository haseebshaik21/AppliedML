import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

# Function to read CSV data
def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to convert all columns to strings for both datasets
def convert_to_str(data):
    return data.astype(str)

# Function to apply one-hot encoding to all fields
def one_hot_encoding(train_data, dev_data):
    onehot_transformer = OneHotEncoder(handle_unknown='ignore')
    X_train_transformed = onehot_transformer.fit_transform(train_data.drop('SalePrice', axis=1))
    X_dev_transformed = onehot_transformer.transform(dev_data.drop('SalePrice', axis=1))
    return X_train_transformed, X_dev_transformed, onehot_transformer

# Function to log-transform the SalePrice
def log_transform_sale_price(train_data, dev_data):
    y_train_log = np.log(train_data['SalePrice'])
    y_dev_log = np.log(dev_data['SalePrice'])
    return y_train_log, y_dev_log

# Function to fit linear regression model and make predictions
def train_and_predict(X_train_transformed, y_train_log, X_dev_transformed):
    model = LinearRegression()
    model.fit(X_train_transformed, y_train_log)
    y_pred_log = model.predict(X_dev_transformed)
    return model, y_pred_log

# Function to calculate RMSLE
def calculate_rmsle(y_dev_log, y_pred):
    rmsle = np.sqrt(mean_squared_log_error(y_dev_log, np.log(y_pred)))
    return rmsle

# Function to get top positive and negative features
def get_top_features(model, onehot_transformer):
    coefficients = model.coef_
    feature_names = onehot_transformer.get_feature_names_out()
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    top_positive_features = feature_importance.sort_values(by='Coefficient', ascending=False).head(10)
    top_negative_features = feature_importance.sort_values(by='Coefficient', ascending=True).head(10)
    
    return top_positive_features, top_negative_features

# Function to preprocess and predict on test data
def preprocess_and_predict(train_data, test_data):
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    train_data[categorical_cols] = train_data[categorical_cols].astype(str)
    test_data[categorical_cols] = test_data[categorical_cols].astype(str)
    
    onehot_transformer = OneHotEncoder(handle_unknown='ignore')
    X_train_transformed = onehot_transformer.fit_transform(train_data.drop(['SalePrice', 'Id'], axis=1))
    X_test_transformed = onehot_transformer.transform(test_data.drop('Id', axis=1))
    
    y_train_log = np.log(train_data['SalePrice'])
    
    model = LinearRegression()
    model.fit(X_train_transformed, y_train_log)
    
    y_test_pred_log = model.predict(X_test_transformed)
    y_test_pred = np.exp(y_test_pred_log)
    
    return y_test_pred, test_data['Id']

# Read data
train_data = read_csv('/content/my_train.csv')
dev_data = read_csv('/content/my_dev.csv')

# Preprocess training and dev data
train_data_str = convert_to_str(train_data)
dev_data_str = convert_to_str(dev_data)

X_train_transformed, X_dev_transformed, onehot_transformer = one_hot_encoding(train_data_str, dev_data_str)
y_train_log, y_dev_log = log_transform_sale_price(train_data, dev_data)

# Train model and predict on dev data
model, y_pred_log = train_and_predict(X_train_transformed, y_train_log, X_dev_transformed)
rmsle = calculate_rmsle(y_dev_log, np.exp(y_pred_log))

# Get top positive and negative features
top_positive_features, top_negative_features = get_top_features(model, onehot_transformer)

# Read test data
test_data = read_csv('/content/test.csv')

# Preprocess test data and make predictions
y_test_pred, test_ids = preprocess_and_predict(train_data, test_data)

# Create submission dataframe
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_test_pred
})

submission.head()
