# Model inference/prediction script
import numpy as np
import tensorflow as tf

# Load test data
X_test = np.load("../data/X_test_FD001.npy")

# Load model
model = tf.keras.models.load_model("../models/best_lstm.h5")

# Predict RUL
y_pred = model.predict(X_test)
print("Predicted RULs:", y_pred.flatten())