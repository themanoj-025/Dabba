"""Helper script to set up Kaggle credentials and download datasets.

Usage:
    1. Go to https://www.kaggle.com/settings/account
    2. Click "Create New Token" — downloads kaggle.json
    3. Place kaggle.json in the same directory as this script
    4. Run: python setup_kaggle.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Set up Kaggle credentials and download datasets."""
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Step 1: Find kaggle.json
    local_json = Path("kaggle.json")
    if not local_json.exists():
        local_json = Path(__file__).parent / "kaggle.json"

    if not local_json.exists():
        print("[ERROR] kaggle.json not found!")
        print()
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print('2. Click "Create New Token" -- this downloads kaggle.json')
        print(f"3. Place kaggle.json in: {kaggle_dir}")
        print(f"   OR place it in: {Path.cwd()}")
        print("4. Run this script again: python setup_kaggle.py")
        sys.exit(1)

    # Step 2: Validate JSON
    try:
        data = json.loads(local_json.read_text())
        if "username" not in data or "key" not in data:
            print("[ERROR] kaggle.json is missing 'username' or 'key' fields")
            sys.exit(1)
        print(f"[OK] Found kaggle.json for user: {data['username']}")
    except json.JSONDecodeError:
        print("[ERROR] kaggle.json is not valid JSON")
        sys.exit(1)

    # Step 3: Copy to ~/.kaggle/
    kaggle_dir.mkdir(exist_ok=True)
    target = kaggle_dir / "kaggle.json"
    shutil.copy2(local_json, target)
    print(f"[OK] Copied kaggle.json to {target}")

    # Step 4: Verify authentication
    print("\n--- Verifying Kaggle authentication ---")
    result = subprocess.run(
        [sys.executable, "-c",
         "from kaggle.api.kaggle_api_extended import KaggleApi; "
         "api = KaggleApi(); api.authenticate(); print('[OK] Kaggle authenticated successfully')"],
        capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"[ERROR] Authentication failed: {result.stderr.strip()}")
        sys.exit(1)

    # Step 5: Download datasets
    print("\n--- Downloading datasets ---")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("himanshupoddar/zomato-bangalore-restaurants", "zomato"),
        ("rajatkumar30/food-delivery-time", "delivery"),
    ]

    for slug, label in datasets:
        print(f"\nDownloading {label} dataset ({slug})...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug,
             "-p", str(raw_dir), "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[OK] {label} dataset downloaded successfully")
        else:
            print(f"[WARN] {label} download issue: {result.stderr.strip()}")

    # Step 6: Verify files
    print("\n--- Verifying downloaded files ---")
    csv_files = list(raw_dir.glob("*.csv"))
    if csv_files:
        for f in csv_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  [FILE] {f.name} ({size_mb:.1f} MB)")
    else:
        print("[ERROR] No CSV files found in data/raw/")
        print("You may need to download manually from Kaggle")

    print("\n--- Done! ---")
    print("Next step: Run 'make train' to execute the full pipeline.")


if __name__ == "__main__":
    main()
