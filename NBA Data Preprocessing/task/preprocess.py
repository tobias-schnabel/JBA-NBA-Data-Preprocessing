import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


def clean_data(path):
    df = pd.read_csv(path)

    # Date and Draft Year Parsing
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y', errors='coerce')

    # Missing Value Imputation
    df['team'].fillna('No Team', inplace=True)

    # Height Conversion
    df['height'] = df['height'].apply(str)
    df['height'] = df['height'].apply(lambda x: float(x.split('/')[1].strip()) if '/' in x else x)

    # Weight Conversion
    df['weight'] = df['weight'].apply(str)
    df['weight'] = df['weight'].apply(lambda x: float(x.split('/')[1].split()[0].strip()) if '/' in x else x)

    # Salary Conversion
    df['salary'] = df['salary'].str.replace('$', '').astype('float')

    # Country Categorization
    df['country'] = df['country'].apply(lambda x: 'USA' if x == 'USA' else 'Not-USA')

    # Draft Round Cleanup
    df['draft_round'].replace('Undrafted', '0', inplace=True)

    return df


def feature_data(df):
    # Extract and parse year from 'version'
    df['version'] = df['version'].apply(lambda x: x[-2:]).astype(int)
    df['version'] = df['version'].apply(lambda x: 2000 + x if x < 100 else x)

    # Engineer 'age' feature
    df['age'] = df['version'] - df['b_day'].dt.year

    # Engineer 'experience' feature
    df['experience'] = df['version'] - df['draft_year'].dt.year

    # Engineer 'bmi' feature
    df['bmi'] = df['weight'] / (df['height'] ** 2)

    # Drop specified columns
    df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True)

    # Remove high cardinality features
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() >= 50:
            df.drop(columns=[col], inplace=True)

    return df


def multicol_data(df):
    # Filter only numerical columns for correlation matrix calculation
    numerical_df = df.select_dtypes(include=[np.number])

    # Calculate the correlation matrix for numerical features
    corr_matrix = numerical_df.corr()

    # Identify pairs of features with high correlation
    high_corr_pairs = []
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i < j and np.abs(corr_matrix.loc[i, j]) > 0.5:
                high_corr_pairs.append((i, j))

    # Find the feature with the lowest absolute correlation with the target variable
    lowest_corr_with_target = None
    lowest_corr_value = float('inf')

    for feature1, feature2 in high_corr_pairs:
        corr_with_target_feature1 = np.abs(corr_matrix['salary'][feature1])
        corr_with_target_feature2 = np.abs(corr_matrix['salary'][feature2])

        if corr_with_target_feature1 < lowest_corr_value:
            lowest_corr_value = corr_with_target_feature1
            lowest_corr_with_target = feature1

        if corr_with_target_feature2 < lowest_corr_value:
            lowest_corr_value = corr_with_target_feature2
            lowest_corr_with_target = feature2

    # Drop the feature with the lowest correlation with the target
    if lowest_corr_with_target:
        df = df.drop(columns=[lowest_corr_with_target])

    return df


def transform_data(df):
    # Separate the target variable and features
    y = df['salary']
    X = df.drop(columns=['salary'])

    # Define the categorical columns that need to be one-hot encoded
    categorical_columns = ['team', 'position', 'country', 'draft_round']

    # Define the numerical columns (all columns of number type except 'salary')
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()

    # Ensure 'draft_round' is treated as a categorical column, not numerical
    if 'draft_round' in numerical_columns:
        numerical_columns.remove('draft_round')

    # Initialize OneHotEncoder separately to get the category names
    enc = OneHotEncoder(sparse=False)
    enc.fit(df[categorical_columns])

    # Get the names of the one-hot encoded columns
    one_hot_feature_names = np.concatenate(enc.categories_).ravel().tolist()

    # Create a ColumnTransformer to apply the transformations to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', enc, categorical_columns)
        ])

    # Apply the transformations
    X_processed = preprocessor.fit_transform(X)

    # Combine the numerical and one-hot encoded column names
    transformed_column_names = numerical_columns + one_hot_feature_names

    # Create a DataFrame with the transformed data and correct column names
    X_transformed = pd.DataFrame(X_processed, columns=transformed_column_names)

    return X_transformed, y


