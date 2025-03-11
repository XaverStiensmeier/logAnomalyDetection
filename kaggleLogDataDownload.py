import kagglehub

# Download latest version
path = kagglehub.dataset_download("krishd123/log-data-for-anomaly-detection")

print("Path to dataset files:", path)