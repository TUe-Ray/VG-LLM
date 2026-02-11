import json

# Load the JSON file
with open('spar_234k.json', 'r') as f:
    data = json.load(f)

# Count by 'id'
if isinstance(data, list):
    count_by_id = len(data)
    print(f"Number of items: {count_by_id}")
    
    # Check for unique IDs
    if data and 'id' in data[0]:
        ids = [item['id'] for item in data if 'id' in item]
        unique_ids = len(set(ids))
        print(f"Unique IDs: {unique_ids}")
        
        # Check for missing IDs
        missing_ids = len([item for item in data if 'id' not in item])
        print(f"Items missing 'id': {missing_ids}")
        
        # Check for duplicate IDs
        duplicate_ids = len(ids) - unique_ids
        print(f"Duplicate IDs: {duplicate_ids}")
else:
    print("Data is not a list.")