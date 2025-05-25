import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load test data
test_df = pd.read_csv("data/test_FD001.txt", sep='\s+', header=None)
cols = ['unit', 'cycle'] + [f'os_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
test_df.columns = cols

# Normalize (ideally, use the scaler from your training, but here for demo we'll just fit new)
feat_cols = [col for col in test_df.columns if col not in ['unit', 'cycle']]
scaler = MinMaxScaler()
test_df[feat_cols] = scaler.fit_transform(test_df[feat_cols])  # NOTE: best to use your train-fitted scaler

# Select an engine/unit to test
unit_id = 1  # change to any unit in test set
seq_length = 30
unit_df = test_df[test_df['unit'] == unit_id]
window = unit_df.iloc[-seq_length:][feat_cols].values  # last 30 cycles
window = window.reshape(1, seq_length, -1)  # shape: (1, 30, features)

# Save for Streamlit app
np.save("test_window.npy", window)
print("Saved test_window.npy with shape:", window.shape)