"""
Semi-supervised log anomaly detection using sentence vector embeddings and a sliding window
"""

import matplotlib.pyplot as plt
import yaml
import os
import time
import datetime
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, CosineSimilarity
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)
from tensorflow.python.client import device_lib

SEED = 42

# Set seed for NumPy
np.random.seed(SEED)

# Set seed for TensorFlow
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

start_time = time.time()

device_lib.list_local_devices()

MODEL_FILENAME = "model.keras"
MODEL_PATH = None # 'outputs/results/experiment_2025-05-08_14-24-52/model.keras'
WINDOW_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.001
LOSS_FUNCTION = 'mse'
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
METRICS = [MeanAbsoluteError(), CosineSimilarity(axis=-1)]

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

rest_data, training_data = train_test_split(normal_data, train_size=0.2, shuffle=False)
validation_data, test_data_normal = train_test_split(rest_data, test_size=0.3, shuffle=False)

print("Test data Total:", len(test_data_normal)+len(test_data_abnormal))

test_data_normal = test_data_normal.head(len(test_data_abnormal))
test_data_abnormal = test_data_abnormal.head(len(test_data_normal))

# test_data = pd.concat([test_data_normal, test_data_abnormal])

def get_number_of_windows(windowed_dataset):
    count = 0
    for _ in windowed_dataset:
        count += 1
    return count

def create_window_tf_dataset(dataset):
    embeddings_list = dataset["embeddings"].tolist()
    embeddings_array = np.array(embeddings_list)
    embedding_tensor = tf.convert_to_tensor(embeddings_array)
    df_tensor  = tf.convert_to_tensor(embedding_tensor)
    tensor_dataset = tf.data.Dataset.from_tensor_slices(df_tensor)
    windowed_dataset = tensor_dataset.window(WINDOW_SIZE, shift=WINDOW_SIZE//2, drop_remainder=True)
    windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(WINDOW_SIZE))
    return windowed_dataset

logging.info(f"Training {training_data.shape}")
logging.info(f"Validation {validation_data.shape}")

num_windows = sum(1 for _ in create_window_tf_dataset(training_data))
print(f"Number of windows created: {num_windows}")

training_data = create_window_tf_dataset(training_data).map(lambda window: (window, window)) # training data is normal
validation_data = create_window_tf_dataset(validation_data).map(lambda window: (window, window)) # training data is normal
test_data_normal = create_window_tf_dataset(test_data_normal).map(lambda window: (window, window)) # normal training data is normal
test_data_abnormal = create_window_tf_dataset(test_data_abnormal).map(lambda window: (window, window)) # abnormal training data is abnormal

# group normal and abnormal data
test_data = test_data_normal.concatenate(test_data_abnormal)
print("n windows test normal", get_number_of_windows(test_data_normal))
print("n windows test abnormal", get_number_of_windows(test_data_abnormal))
print("n windows test concatenated", get_number_of_windows(test_data))
# iterator = iter(test_data)
# first_entry = next(iterator)
# print("Should be 2 window of embeddings and label:", len(first_entry))
# print(f"Should be window [size, length of embedding vector] []{WINDOW_SIZE}, {embedding_length}]:", len(first_entry[0].shape))

training_data = training_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)
for x, y in training_data.take(1):
    print("Input shape:", x.shape)
    print("Label shape:", y.shape)
train_batches = sum(1 for _ in training_data)
print(f"Counted training batches: {train_batches}")
#exit(0)

if not MODEL_PATH:

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
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS) # accuracy doesn't make sense use cosine similarity

    # Train the model
    history = model.fit(training_data, epochs=EPOCHS, validation_data=validation_data)

    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"Elapsed time {elapsed_time}")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = f'outputs/results/experiment_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    plot_model(model, to_file=os.path.join(results_dir, 'model.png'), show_shapes=True, show_layer_names=True)

    def save_plot(metric_name, title, ylabel, legend_loc='upper right'):
        plt.figure()
        plt.plot(history.history[metric_name])
        plt.plot(history.history['val_' + metric_name])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc=legend_loc)
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'{metric_name}.png'))
        plt.close()

    save_plot('loss', 'MSE Loss', 'Loss')
    save_plot('mean_absolute_error', 'Mean Absolute Error', 'MAE')
    save_plot('cosine_similarity', 'Cosine Similarity', 'Cosine Similarity', legend_loc='lower right')

    model.save(os.path.join(results_dir, MODEL_FILENAME))

    with open(os.path.join(results_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    params = {
        'timestamp': timestamp,
        'loss_function': LOSS_FUNCTION,
        'metrics': str(METRICS),
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'optimizer': str(OPTIMIZER),
        'model_file': MODEL_FILENAME,
    }
    with open(os.path.join(results_dir, 'params.yaml'), 'w') as f:
        yaml.dump(params, f)
else:
    logging.info(f"Skipping Model Training. Loading model {MODEL_PATH}.")
    model = tf.keras.models.load_model(MODEL_PATH)

def calculate_reconstruction_errors(dataset, model, name=""):
    """
    Calculate reconstruction errors for the given dataset using the provided model.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to evaluate.
        model (tf.keras.Model): The trained autoencoder model used for reconstruction.

    Returns:
        list: A list of reconstruction errors for each input in the dataset.
    """
    reconstruction_errors = []

    # Iterate over all batches in the dataset
    for inputs, _ in dataset:
        print(len(inputs))
        # Predict the reconstruction for each batch
        reconstructed = model.predict(inputs)

        # Calculate mean squared errors for the batch and accumulate them
        batch_errors = tf.reduce_mean(tf.square(reconstructed - inputs), axis=[1, 2])
        reconstruction_errors.extend(batch_errors.numpy())

    print(name, "Error Average", np.mean(reconstruction_errors))
    return reconstruction_errors

def detect_anomalies(errors, threshold):
    return (errors > threshold).astype(int)

def evaluate(test_labels, predictions, name="", full=False):
    print(f"-- {name} --", "\n", predictions)
    assert len(predictions) == len(test_labels)
    
    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:\n", cm)
    if full:
        mcc = matthews_corrcoef(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=1)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, test_errors)
        prc_auc = average_precision_score(test_labels, test_errors)
        print(f"Matthews Correlation Coefficient: {mcc:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"ROC AUC Score: {roc_auc:.2f}")
        print(f"PRC AUC Score: {prc_auc:.2f}")

# Calculate reconstruction errors
training_errors = calculate_reconstruction_errors(training_data, model, name="Training")
validation_errors = calculate_reconstruction_errors(validation_data, model, name="Validation")
test_errors = calculate_reconstruction_errors(test_data, model, name="Test") # test_data_normal, test_data_abnormal
test_normal_errors = calculate_reconstruction_errors(test_data_normal.batch(BATCH_SIZE), model, name="Normal Test")
test_abnormal_errors = calculate_reconstruction_errors(test_data_abnormal.batch(BATCH_SIZE), model, name="Abnormal Test")

threshold = np.percentile(validation_errors, 95)
print(f"Computed threshold at 95th percentile: {threshold}")

# Detect anomalies test
predicted_labels1 = detect_anomalies(test_errors, threshold)
half = len(predicted_labels1) // 2
test_labels1 = [0] * half + [1]*(half)
evaluate(test_labels1, predicted_labels1, name="Test", full=True)

# Detect anomalies test_normal
predicted_labels2 = detect_anomalies(test_normal_errors, threshold)
test_labels2 = [0]*len(predicted_labels2)
evaluate(test_labels2, predicted_labels2, name="Test Normal")

# Detect anomalies test_abnormal
predicted_labels3 = detect_anomalies(test_abnormal_errors, threshold)
test_labels3 = [0]*len(predicted_labels3)
evaluate(test_labels3, predicted_labels3, name="Test Abnormal")

# Detect anomalies training
predicted_labels4 = detect_anomalies(training_errors, threshold)
test_labels4 = [0]*len(predicted_labels4)
evaluate(test_labels4, predicted_labels4, name="Training")