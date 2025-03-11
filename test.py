import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s", 
    level=logging.DEBUG, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Step 1: Load and prepare data
input_pickle = "outputs/parsed_openstack_logs_with_embeddings_final.pickle"
with open(input_pickle, "rb") as handle:
    df = pickle.load(handle)

logging.info("DataFrame loaded successfully.")
logging.debug(f"\n{df.head()}")

# Prepare data
normal_data = df[df['label'] == 0]['embeddings'].values
print(len(normal_data[0]))
exit(0)

anomalous_data = df[df['label'] == 1]['embeddings'].values

X_normal = np.array([np.array(x) for x in normal_data])
X_anomalous = np.array([np.array(x) for x in anomalous_data])
X_train, X_val, _, _ = train_test_split(X_normal, X_normal, test_size=0.5, random_state=42)
X_val, X_test, _, _ = train_test_split(X_val, X_val, test_size=0.5, random_state=42)

# Convert to PyTorch tensors and reshape to fit LSTM
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

# Logging the shapes
logging.info(f"Training data shape: {X_train_tensor.shape}")
logging.info(f"Validation data shape: {X_val_tensor.shape}")
logging.info(f"Test data shape: {X_test_tensor.shape}")

# Encoder Class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.lstm_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        _, (last_h_state, _) = self.lstm_enc(x)
        x_enc = last_h_state[-1]
        return x_enc

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.lstm_dec = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, z, seq_len):
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.lstm_dec(z)
        dec_out = self.fc(dec_out)
        return dec_out

# LSTM Auto-Encoder Class
class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio, seq_len):
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio)
        self.seq_len = seq_len
        
    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc, self.seq_len)
        return x_dec

# Model Parameters
input_dim = X_train_tensor.shape[2]
hidden_dim = 128
latent_dim = 32
seq_len = X_train_tensor.shape[1]
dropout_ratio = 0.2

# Instantiate the model
model = LSTMAE(input_size=input_dim, hidden_size=latent_dim, dropout_ratio=dropout_ratio, seq_len=seq_len)
logging.info("Model Architecture:")
logging.info(model)

# Step 4: Train the model on normal data
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    reconstructed = model(X_train_tensor)
    loss = criterion(reconstructed, X_train_tensor) * 100
    
    loss.backward()
    optimizer.step()
    
    logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Step 5: Evaluate the model
model.eval()
with torch.no_grad():
    reconstructed_val = model(X_val_tensor)
    val_loss = torch.mean((reconstructed_val - X_val_tensor) ** 2, dim=[1, 2]).numpy()
    logging.info(f"Val Loss: {val_loss}")
    
    reconstructed_test = model(X_test_tensor)
    test_loss = torch.mean((reconstructed_test - X_test_tensor) ** 2, dim=[1, 2]).numpy()
    logging.info(f"Test Loss: {test_loss}")

# Step 6: Anomaly Detection
threshold = np.percentile(val_loss, 95)
while threshold:
    logging.info(f"Anomaly detection threshold: {threshold:.6f}")
    y_val_pred = (val_loss > threshold).astype(int)
    y_test_pred = (test_loss > threshold).astype(int)
    
    print(y_value_true)

    # Step 7: Evaluation
    y_val_true = np.zeros_like(y_val_pred)
    y_test_true = np.concatenate([np.zeros_like(y_test_pred[:len(X_val)]), np.ones_like(y_test_pred[len(X_val):])])
    logging.info("Confusion Matrix (Validation):")
    logging.info(f"\n{confusion_matrix(y_val_true, y_val_pred)}")
    logging.info("Confusion Matrix (Test):")
    logging.info(f"\n{confusion_matrix(y_test_true, y_test_pred)}")
    logging.info("Evaluation Metrics (Test):")
    logging.info(f"Precision: {precision_score(y_test_true, y_test_pred):.4f}")
    logging.info(f"Recall: {recall_score(y_test_true, y_test_pred):.4f}")
    logging.info(f"F1 Score: {f1_score(y_test_true, y_test_pred):.4f}")
    logging.info(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test_true, y_test_pred):.4f}")
    threshold = float(input("New threshold: "))