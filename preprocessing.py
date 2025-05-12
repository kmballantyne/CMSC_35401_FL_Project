import pandas as pd

# Load CSV
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")

# Extract transformation category
def get_transform_category(path):
    
    '''
    Extracts the transformation category from the file path.
    Args:
        path (str): The file path.'''
        
    if 'synthetic/digital' in path:
        return 'digital'
    elif 'synthetic/photographic' in path:
        return 'photographic'
    else:
        return 'unknown'

train_df['TransformCategory'] = train_df['Path'].apply(get_transform_category)
valid_df['TransformCategory'] = valid_df['Path'].apply(get_transform_category)

# ✅ Only keep digital and photographic (ignore original/natural)
subset_df = train_df[train_df['TransformCategory'].isin(['digital', 'photographic'])]

# Sample 2500 from each
sampled_digital = subset_df[subset_df['TransformCategory'] == 'digital'].sample(n=2500, random_state=42)
sampled_photo = subset_df[subset_df['TransformCategory'] == 'photographic'].sample(n=2500, random_state=42)

# Split into 2 clients each (1000 + 1500 to simulate imbalance or evenly)
clients = {}

# Digital → clients 0 and 1
clients[0] = sampled_digital.iloc[:1000].copy()
clients[0]['client_id'] = 0

clients[1] = sampled_digital.iloc[1000:2000].copy()
clients[1]['client_id'] = 1

# Photographic → clients 2 and 3
clients[2] = sampled_photo.iloc[:1000].copy()
clients[2]['client_id'] = 2

clients[3] = sampled_photo.iloc[1000:2000].copy()
clients[3]['client_id'] = 3

# Mixed client → leftovers from both
remaining = pd.concat([
    sampled_digital.iloc[2000:], 
    sampled_photo.iloc[2000:]
], ignore_index=True).sample(n=1000, random_state=99)
remaining['client_id'] = 4
clients[4] = remaining

# Show metadata per client
# for i in range(5):
#     client_df = clients[i]
#     print(f"Client {i} has {len(client_df)} samples")
#     print(client_df.head())
#     print(client_df['TransformCategory'].value_counts())
#     print("\n")

# Save metadata per client
for i in range(5):
    client_df = clients[i]
    client_df.to_csv(f"client_{i}_metadata.csv", index=False)
    print(f"✅ Saved client_{i}_metadata.csv with {len(client_df)} samples")

