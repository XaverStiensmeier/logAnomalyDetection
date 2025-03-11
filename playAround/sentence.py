from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define some test sentences
sentences = [
    "The server is down due to an error.",
    "An error caused the server to crash.",
    "User logged in successfully.",
    "System reboot detected.",
    "The server has failed due to unforeseen circumstances! And potatoes!"
]

# Generate embeddings
embeddings = model.encode(sentences)

# Compute cosine similarity between sentence pairs
similarity_matrix = cosine_similarity(embeddings)

# Print the similarity matrix
print("Cosine Similarity Matrix:")
print(similarity_matrix)