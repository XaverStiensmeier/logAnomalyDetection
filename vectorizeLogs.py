import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

PLOT = True
CSV = False
input_csv = "outputs/parsed_openstack_logs.csv"
output_csv = "outputs/parsed_openstack_logs_with_embeddings.csv"
output_pickle = "outputs/parsed_openstack_logs_with_embeddings.pickle"

# Load the parsed logs data
df = pd.read_csv(output_csv)

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vectorize the messages
embeddings = model.encode(df['message'].tolist(), show_progress_bar=True)
print("Shape of embeddings:", embeddings.shape)
# Add the embeddings to the DataFrame
# We're using a 2D NumPy array, so we need to handle how it's stored in the DataFrame
df['embeddings'] = embeddings.tolist()

# Save the dataframe with embeddings to a new CSV file, or you can continue to use it in memory
if CSV:
    df.to_csv(output_csv, index=False)
else: # pickle
    print("Saving pickle")
    with open(output_pickle, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Vectorization complete! Embeddings added to DataFrame. Saved at", output_csv if CSV else output_pickle)


if PLOT:
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot the first two principal components
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.title('PCA of Log Message Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

