import os
import pandas as pd
from drain3 import TemplateMiner

# Define paths
base_path = "/home/xaver/.cache/kagglehub/datasets/krishd123/log-data-for-anomaly-detection/versions/2"
openstack_path = os.path.join(base_path, "OpenStack_log", "OpenStack_log")
print(f"Assuming path {openstack_path}...")

# Log file paths
log_files = {
    "normal1": os.path.join(openstack_path, "openstack_normal1.log"),
    "normal2": os.path.join(openstack_path, "openstack_normal2.log"),
    "abnormal": os.path.join(openstack_path, "openstack_abnormal.log"),
}

# Initialize Drain3
template_miner = TemplateMiner()

# Process logs
parsed_logs = []
for label, file_path in log_files.items():
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed_result = template_miner.add_log_message(line.strip())
            log_key = parsed_result["cluster_id"]  # Log template ID
            parsed_logs.append({"log_key": log_key, "message": line.strip(), "label": label})

# Convert to Pandas DataFrame
df = pd.DataFrame(parsed_logs)

# Save parsed logs for later use
save_path = "outputs/parsed_openstack_logs.csv"
df.to_csv(save_path, index=False)
print(f"Parsing complete! Saved to '{save_path}'.")