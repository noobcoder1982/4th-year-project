from huggingface_hub import hf_hub_download
import pandas as pd

print("Getting dataset...")

file_path = hf_hub_download(
    repo_id="Yogeshwaran10/fact-verification-dataset",
    filename="data/train-00000-of-00001.parquet",
    repo_type="dataset"
)

print("\nDataset file:")
print(file_path)

print("\nReading dataset...")

df = pd.read_parquet(file_path)

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nFirst 5 examples:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes)