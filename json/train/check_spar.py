import json
import os
from pathlib import Path
import random

def check_spar_data(json_file="spar_234k.json", data_dir=None, sample_size=None, seed=42):
    """
    Check if all data files mentioned in json_file exist in data_dir.
    
    Args:
        json_file: Path to the JSON file containing data references
        data_dir: Directory containing SPAR data (default: $SCRATCH/spar_workspace/spar_data)
        sample_size: If set, randomly check N items instead of all (default: None = check all)
        seed: Random seed for reproducibility
    """
    
    if data_dir is None:
        data_dir = os.path.join(os.environ.get("SCRATCH", "."), "spar_workspace", "spar_data")
    
    data_dir = Path(data_dir)
    
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print("Error: JSON file should contain a list of items")
        return
    
    # Sample if requested
    items_to_check = data
    if sample_size:
        random.seed(seed)
        items_to_check = random.sample(data, min(sample_size, len(data)))
    
    missing_files = []
    found_count = 0
    
    print(f"Checking {len(items_to_check)}/{len(data)} items in {data_dir}")
    print("-" * 60)
    
    for item in items_to_check:
        # Adjust based on your JSON structure (this is a common pattern)
        file_path = item.get("file") or item.get("path") or item.get("image")
        
        if not file_path:
            print(f"Warning: No file path found in item: {item}")
            continue
        
        full_path = data_dir / file_path
        
        if full_path.exists():
            found_count += 1
        else:
            missing_files.append(file_path)
    
    # Results
    print(f"Found: {found_count}/{len(items_to_check)}")
    print(f"Missing: {len(missing_files)}/{len(items_to_check)}")
    
    if missing_files:
        print("\nMissing files:")
        for f in missing_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return found_count == len(items_to_check)

if __name__ == "__main__":
    # Check all files
    # check_spar_data()
    
    # Or randomly check 100 files
    check_spar_data(sample_size=100)