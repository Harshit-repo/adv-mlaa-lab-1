import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_features(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Filling missing values with mean
        ('scaler', StandardScaler())                 # Normalizing
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Filling missing values with most frequent
        ('onehot', OneHotEncoder(handle_unknown='ignore'))    # One-hot encoding
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    df_processed = preprocessor.fit_transform(df)
    df_processed = pd.DataFrame(df_processed.toarray(), columns=preprocessor.get_feature_names_out())
    if 'drafted' in df.columns:
        df_processed['drafted'] = df['drafted'].values
    df_processed.to_csv(output_filepath, index=False)
    
if __name__ == '__main__':
    build_features('../data/raw/train.csv', '../data/processed/train_processed.csv')