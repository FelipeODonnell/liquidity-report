# Plan for API Data Collection and Storage

This document outlines the plan to modify all files in the 'coinglass-api' folder to save API responses as CSV files in the 'data/coinglass' folder.

## Modification Strategy

### 1. API File Modifications

For each Python file in the `coinglass-api` folder, we'll add code to:
1. Parse the JSON response
2. Convert the data to a pandas DataFrame
3. Save the DataFrame as a CSV file in the corresponding location within the `data/coinglass` folder
4. Preserve all existing code including comments

#### Example modification:

Before:
```python
import requests

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/list"

headers = {
    "accept": "application/json",
    "CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"
}

response = requests.get(url, headers=headers)

print(response.text)

'''
[Documentation/comments...]
'''
```

After:
```python
import requests
import json
import pandas as pd
import os
from datetime import datetime

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/list"

headers = {
    "accept": "application/json",
    "CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"
}

response = requests.get(url, headers=headers)

print(response.text)

# Save response data as CSV
try:
    # Parse JSON response
    response_data = json.loads(response.text)
    
    # Check if response contains data
    if response_data.get('code') == '0' and 'data' in response_data:
        # Convert to DataFrame
        df = pd.DataFrame(response_data['data'])
        
        # Create directory if it doesn't exist
        os.makedirs('data/coinglass/etf', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f'data/coinglass/etf/api_etf_bitcoin_list_{timestamp}.csv'
        
        # Save as CSV
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
except Exception as e:
    print(f"Error saving data: {e}")

'''
[Documentation/comments...]
'''
```

### 2. Implementation Steps

1. Add necessary imports to each file:
   - `import json`
   - `import pandas as pd`
   - `import os`
   - `from datetime import datetime`

2. Add data saving code after the API response but before any documentation comments.

3. Handle nested JSON structures appropriately:
   - For simple structures: directly convert to DataFrame
   - For complex nested structures: flatten or extract relevant portions

4. Create directories as needed to match the structure in `coinglass-api/`.

5. Use consistent naming convention: `{original_filename}_{timestamp}.csv`

## report.py Implementation

The `report.py` file will:
1. Scan all API files in the `coinglass-api` directory
2. Present a menu for users to select which API endpoints to run
3. Execute selected files and confirm data saved

### Proposed structure:

```python
import os
import subprocess
import sys
from typing import List, Dict, Tuple

def get_all_api_files() -> List[Tuple[str, str]]:
    """
    Scan the coinglass-api directory and return all API files with their categories.
    Returns a list of tuples: (file_path, category)
    """
    api_files = []
    root_dir = 'coinglass-api'
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        category = os.path.relpath(dirpath, root_dir)
        if category == '.':
            category = 'root'
            
        for filename in filenames:
            if filename.endswith('.py') and filename.startswith('api_'):
                file_path = os.path.join(dirpath, filename)
                api_files.append((file_path, category))
    
    return api_files

def display_menu(api_files: List[Tuple[str, str]]) -> None:
    """Display the menu of available API files grouped by category."""
    categories = {}
    
    # Group files by category
    for file_path, category in api_files:
        if category not in categories:
            categories[category] = []
        categories[category].append(file_path)
    
    # Display menu
    print("\n=== Coinglass API Data Collection Tool ===\n")
    print("Available API endpoints by category:\n")
    
    file_index = 1
    for category, files in categories.items():
        print(f"\n--- {category} ---")
        for file_path in files:
            filename = os.path.basename(file_path)
            print(f"{file_index}. {filename}")
            file_index += 1
    
    print("\nOptions:")
    print("A. Run all API endpoints")
    print("Q. Quit")

def run_selected_files(api_files: List[Tuple[str, str]], selections: List[int]) -> None:
    """Run the selected API files."""
    for idx in selections:
        if 1 <= idx <= len(api_files):
            file_path, _ = api_files[idx-1]
            print(f"\nRunning: {file_path}")
            try:
                subprocess.run([sys.executable, file_path], check=True)
                print(f"Completed: {file_path}")
            except subprocess.CalledProcessError:
                print(f"Error running: {file_path}")

def main():
    api_files = get_all_api_files()
    
    while True:
        display_menu(api_files)
        
        choice = input("\nEnter selection (comma-separated numbers, 'A' for all, or 'Q' to quit): ").strip().upper()
        
        if choice == 'Q':
            print("Exiting...")
            break
        
        elif choice == 'A':
            print("\nRunning all API endpoints...")
            run_selected_files(api_files, range(1, len(api_files) + 1))
        
        else:
            try:
                # Parse comma-separated selections
                selections = [int(x.strip()) for x in choice.split(',')]
                run_selected_files(api_files, selections)
            except ValueError:
                print("Invalid selection. Please enter numbers separated by commas, 'A', or 'Q'.")

if __name__ == "__main__":
    main()
```

## Execution Plan

1. **Initial Setup**:
   - Ensure all required libraries are installed (pandas, requests)
   - Verify the `data/coinglass` directory exists or can be created

2. **File Modification Process**:
   - Create a script to automate the modification of all API files
   - Test on a small subset before applying to all files
   - Apply changes to all files

3. **Testing**:
   - Test individual API file modifications to ensure data is saved correctly
   - Verify directory structure is maintained
   - Check CSV file content and format

4. **report.py Implementation**:
   - Develop and test the report.py file
   - Ensure it correctly identifies and presents all API endpoints
   - Test execution of selected endpoints

5. **Documentation**:
   - Update README.md with information about the new functionality
   - Document usage of report.py

## Notes

- All original code and comments will be preserved
- Only adding functionality to save the data as CSV
- The directory structure in `data/coinglass` will mirror the structure in `coinglass-api/`
- Files will include timestamps to avoid overwriting previous data