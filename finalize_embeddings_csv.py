import pandas as pd
import pickle
from ast import literal_eval

# Load the dataset
input_csv = "outputs/parsed_openstack_logs_with_embeddings.csv"
input_pickle = "outputs/parsed_openstack_logs_with_embeddings.pickle"
output_csv = "outputs/parsed_openstack_logs_with_embeddings_final.csv"
output_pickle = "outputs/parsed_openstack_logs_with_embeddings_final.pickle"

CSV = False

# Read the CSV file
if CSV:
    df = pd.read_csv(input_csv, usecols=["label", "embeddings"], converters={'embeddings': literal_eval})  # "log_key", 
else: # pickle
    with open(input_pickle, 'rb') as handle:
        df = pickle.load(handle)[["label", "embeddings"]]


# Map labels: 'normal1' and 'normal2' to 0, 'abnormal' to 1
df["label"] = df["label"].apply(lambda x: 0 if x.startswith("normal") else 1)

# Save the processed data
df.to_csv(output_csv, index=False)

with open(output_pickle, 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Processing complete. Saved to:", output_csv if CSV else output_pickle)