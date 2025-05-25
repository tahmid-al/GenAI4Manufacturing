# LSTM model training script
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Load preprocessed data
X_train = np.load("../data/X_train_FD001.npy")
y_train = np.load("../data/y_train_FD001.npy")

# Split validation
split = int(0.8 * len(X_train))
X_val, y_val = X_train[split:], y_train[split:]
X_train, y_train = X_train[:split], y_train[:split]

# Build LSTM model
model = models.Sequential([
    layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Callbacks
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mc = callbacks.ModelCheckpoint("../models/best_lstm.h5", save_best_only=True, monitor='val_loss')

# Train
model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[es, mc]
)