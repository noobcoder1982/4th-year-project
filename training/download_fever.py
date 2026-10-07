import requests
from pathlib import Path

DATA_DIR = Path("data/fever")
DATA_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "train.jsonl": "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl",
    "dev.jsonl": "https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl",
    "test.jsonl": "https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl",
}

for filename, url in files.items():

    output = DATA_DIR / filename

    print(f"\nDownloading {filename}...")

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"ERROR: HTTP {response.status_code}")
        continue

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\rProgress: {percent:.1f}%",
                        end=""
                    )

    print(f"\nSaved: {output}")

print("\nFEVER download complete!")