import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def convert_total_charges(df):
    """Convert TotalCharges to numeric and fill missing with 0."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df


def drop_columns(df, columns):
    """Drop specified columns safely."""
    return df.drop(columns=columns, errors='ignore')


def encode_binary_columns(df):
    """Encode Yes/No binary columns as 1/0."""
    binary_cols = [
        col for col in df.columns
        if df[col].nunique() == 2 and df[col].dropna().isin(['Yes', 'No']).all()
    ]
    df[binary_cols] = df[binary_cols].replace({'Yes': 1, 'No': 0})
    return df


def scale_numeric_columns(df, cols_to_scale):
    """Scale numeric columns using StandardScaler."""
    scaler = StandardScaler()
    for col in cols_to_scale:
        if col in df.columns:
            df[[col]] = scaler.fit_transform(df[[col]])
    return df


def full_preprocess_pipeline(df, encoder=None, fit_encoder=False):
    """
    Apply preprocessing steps consistent with training.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        encoder (OneHotEncoder, optional): Pre-fitted encoder.
        fit_encoder (bool): Whether to fit a new encoder.

    Returns:
        If fit_encoder=True: (processed_df, encoder)
        If fit_encoder=False: processed_df only.
    """
    df = df.copy()

    # Convert TotalCharges
    df = convert_total_charges(df)

    # Drop customerID if present
    if 'customerID' in df.columns:
        df = drop_columns(df, ['customerID'])

    # Encode gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Encode binary columns
    df = encode_binary_columns(df)

    # Identify multi-category categorical columns
    cat_cols = [
        c for c in df.select_dtypes(include=['object']).columns
        if df[c].nunique() > 2
    ]

    # Fit or use existing encoder
    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df[cat_cols])
    else:
        if encoder is None:
            raise ValueError('Encoder must be provided when fit_encoder=False')
        encoded = encoder.transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )

    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    # Scale numeric columns
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df = scale_numeric_columns(df, cols_to_scale)

    # Convert any remaining object columns to numeric codes
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category').cat.codes

    if fit_encoder:
        return df, encoder
    return df