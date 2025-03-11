import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

np.random.seed(0)

data_length = 300
data = np.sin(np.linspace(0, 20, data_length)) + np.random.normal(scale=0.5, size=data_length)
# Introduce Anomalies
data[50] += 6
data[150] += 7
data[250] += 8

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

window_size = 10
sequences = create_sequences(data, window_size)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, window_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sequences = torch.tensor(sequences, dtype=torch.float32)

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(sequences)
    loss = criterion(output, sequences)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
with torch.no_grad():
    predictions = model(sequences)
    losses = torch.mean((predictions - sequences)**2, dim=1)
    plt.hist(losses.numpy(), bins=50)
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.show()

# Threshold for defining an anomaly
threshold = losses.mean() + 2 * losses.std()
print(f"Anomaly threshold: {threshold.item()}")

# Detecting anomalies
anomalies = losses > threshold
anomaly_positions = np.where(anomalies.numpy())[0]
print(f"Anomalies found at positions: {np.where(anomalies.numpy())[0]}")

# Plotting anomalies on the time-series graph
plt.figure(figsize=(10, 6))
plt.plot(data, label='Data')
plt.scatter(anomaly_positions, data[anomaly_positions], color='r', label='Anomaly')
plt.title("Time Series Data with Detected Anomalies")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.show()