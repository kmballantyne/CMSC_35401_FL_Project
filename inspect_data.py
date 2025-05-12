import pandas as pd

# Load the CSV files (adjust the path as needed)
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")

# View basic structure
print("\n--- Train CSV Info ---")
print(train_df.info())
# print(train_df.head())
print(train_df.columns)

print("\n--- Unique Transformation Counts ---")
def get_transform_type(path):
    """
    Extracts the transformation type from the file path.
    Args:
        path (str): The file path.
    Returns:
        str: The transformation type (e.g., 'original', 'digital', 'spatial').
    """
    # Split the path by '/' and check the third part
    parts = path.split('/')
    try:
        # For synthetic paths: CheXphoto-v1.0/train/synthetic/{type}/{optional...}
        if parts[2] == 'synthetic':
            return parts[3]  # e.g., 'digital' or 'spatial'
        else:
            return 'original'
    except IndexError:
        return 'unknown'

train_df['TransformCategory'] = train_df['Path'].apply(get_transform_type)
print(train_df['TransformCategory'].value_counts())

# View basic structure
print("\n--- Validation CSV Info ---")
print(valid_df.info())
# print(valid_df.head())
print(valid_df.columns)

valid_df['TransformCategory'] = valid_df['Path'].apply(get_transform_type)
print(valid_df['TransformCategory'].value_counts())