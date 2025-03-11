"""
Semi-supervised log anomaly detection using sentence vector embeddings and a sliding window
"""

import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

WINDOW_SIZE = 100
BATCH_SIZE = 32

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Step 1: Load and prepare data
input_pickle = "outputs/parsed_openstack_logs_with_embeddings_final.pickle"
with open(input_pickle, "rb") as handle:
    df = pickle.load(handle)

logging.info("DataFrame loaded successfully.")
# embeddings will be the input and output since AE tries to create the representation from the latent space
# label contains whether the data is anomalous (1) or normal (0).
logging.debug(f"{df.columns}")
logging.debug(f"\n{df.head()}")

# split data so that training data contains no anomalies, validation and testing contain an equal number of normal and abnormal entries
embedding_length = len(df["embeddings"][0])
print("Length of embedding vector:", embedding_length)
normal_data = df[df['label'] == 0]
test_data_abnormal = df[df['label'] == 1]
abnormal_normal_ratio = len(test_data_abnormal) / len(normal_data)
logging.info(f"N normal: {len(normal_data)}; N abnormal: {len(test_data_abnormal)} -- Ratio: {abnormal_normal_ratio}")

training_data, rest_data = train_test_split(normal_data, train_size=0.8, shuffle=False)
validation_data, test_data_normal = train_test_split(rest_data, test_size=0.3, shuffle=False)

test_data_normal = test_data_normal.head(len(test_data_abnormal))
test_data_abnormal = test_data_abnormal.head(len(test_data_normal))

test_data = pd.concat([test_data_normal, test_data_abnormal])

def create_window_tf_dataset(dataset):
    embeddings_list = dataset["embeddings"].tolist()
    embeddings_array = np.array(embeddings_list)
    embedding_tensor = tf.convert_to_tensor(embeddings_array)
    df_tensor  = tf.convert_to_tensor(embedding_tensor)
    tensor_dataset = tf.data.Dataset.from_tensor_slices(df_tensor)
    windowed_dataset = tensor_dataset.window(WINDOW_SIZE, shift=WINDOW_SIZE, drop_remainder=True)
    windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(WINDOW_SIZE))
    return windowed_dataset

logging.info(f"Training {training_data.shape}")
logging.info(f"Validation {validation_data.shape}")
logging.info(f"Test {test_data.shape}")

training_data = create_window_tf_dataset(training_data).map(lambda window: (window, 0)) # training data is normal
validation_data = create_window_tf_dataset(validation_data).map(lambda window: (window, 0)) # training data is normal
test_data_normal = create_window_tf_dataset(test_data_normal).map(lambda window: (window, 0)) # normal training data is normal
test_data_abnormal = create_window_tf_dataset(test_data_abnormal).map(lambda window: (window, 0)) # abnormal training data is abnormal

# group normal and abnormal data
test_data = test_data_normal.concatenate(test_data_abnormal)
# iterator = iter(test_data)
# first_entry = next(iterator)
# print("Should be 2 window of embeddings and label:", len(first_entry))
# print(f"Should be window [size, length of embedding vector] []{WINDOW_SIZE}, {embedding_length}]:", len(first_entry[0].shape))

training_data = training_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# Build the Autoencoder model
model = models.Sequential([
    Input(shape=(WINDOW_SIZE, embedding_length)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32, return_sequences=False),
    layers.RepeatVector(WINDOW_SIZE),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(64, return_sequences=True),
    layers.TimeDistributed(layers.Dense(embedding_length))
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(training_data, epochs=50, validation_data=validation_data)

# Evaluate on test data
test_batches = list(test_data)  # Convert to list to iterate easily
test_inputs, test_labels = test_batches[0]

# Run prediction
reconstructed = model.predict(test_inputs)

# Calculate reconstruction errors
reconstruction_errors = tf.reduce_mean(tf.square(reconstructed - test_inputs), axis=[1, 2])

# Determine a threshold - for example, use the 95th percentile of reconstruction errors on validation data
# Apply this computation on the validation set to get a consistent threshold
validation_batches = list(validation_data)
val_inputs, _ = validation_batches[0]
val_reconstructed = model.predict(val_inputs)
val_reconstruction_errors = tf.reduce_mean(tf.square(val_reconstructed - val_inputs), axis=[1, 2])
threshold = np.percentile(val_reconstruction_errors, 95)

# Detect anomalies
predicted_labels = (reconstruction_errors > threshold).numpy().astype(int)

# Compute confusion matrix and metrics
cm = confusion_matrix(test_labels, predicted_labels)
mcc = matthews_corrcoef(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, zero_division=1)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)
roc_auc = roc_auc_score(test_labels, reconstruction_errors)
prc_auc = average_precision_score(test_labels, reconstruction_errors)

# Print results
print("Confusion Matrix:\n", cm)
print(f"Matthews Correlation Coefficient: {mcc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print(f"PRC AUC Score: {prc_auc:.2f}")