#!/usr/bin/env python3
"""
Script to modify all Coinglass API files to save responses as CSV files

This script:
1. Scans the coinglass-api directory for all Python API files
2. Adds code to save API responses as CSV files in the data/coinglass directory
3. Preserves all existing code and comments
"""

import os
import re
from typing import List, Tuple


def find_api_files() -> List[str]:
    """Find all Python API files in the coinglass-api directory"""
    api_files = []
    root_dir = 'coinglass-api'
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py') and filename.startswith('api_'):
                file_path = os.path.join(dirpath, filename)
                api_files.append(file_path)
    
    return api_files


def generate_csv_save_code(file_path: str) -> str:
    """Generate the code to save API response as CSV for a specific file"""
    # Extract category (subdirectory) from file path
    parts = file_path.split(os.sep)
    category_path = os.sep.join(parts[1:-1])  # Skip 'coinglass-api' and the filename
    
    # Output directory will mirror the input directory structure
    output_dir = os.path.join('data', 'coinglass', category_path)
    
    # Extract the base filename without extension
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    return f"""
# Save response data as CSV
try:
    # Parse JSON response
    response_data = json.loads(response.text)
    
    # Check if response contains data
    if response_data.get('code') == '0' and 'data' in response_data:
        # Convert to DataFrame
        df = pd.DataFrame(response_data['data'])
        
        # Create directory if it doesn't exist
        os.makedirs('{output_dir}', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f'{output_dir}/{base_filename}_{{timestamp}}.csv'
        
        # Save as CSV
        df.to_csv(file_path, index=False)
        print(f"Data saved to {{file_path}}")
except Exception as e:
    print(f"Error saving data: {{e}}")
"""


def add_imports(content: str) -> str:
    """Add necessary imports if they don't already exist"""
    required_imports = {
        "import json\n": False,
        "import pandas as pd\n": False,
        "import os\n": False,
        "from datetime import datetime\n": False
    }
    
    # Check for existing imports
    for imp in required_imports:
        if imp in content:
            required_imports[imp] = True
    
    # Find where to insert imports
    import_section = ""
    for imp, exists in required_imports.items():
        if not exists:
            import_section += imp
    
    # If we have imports to add, insert them after the existing imports
    if import_section:
        # Try to find the last import statement
        import_pattern = r'^import .*$|^from .* import .*$'
        matches = list(re.finditer(import_pattern, content, re.MULTILINE))
        
        if matches:
            last_import_end = matches[-1].end()
            content = content[:last_import_end] + "\n" + import_section + content[last_import_end:]
        else:
            # If no imports found, add at the beginning
            content = import_section + content
    
    return content


def find_insertion_point(content: str) -> int:
    """Find where to insert the CSV save code"""
    # Try to find the line after the API call but before comments
    response_print_pattern = r'print\(response\.text\)'
    comment_pattern = r"^'''"
    
    response_print_match = re.search(response_print_pattern, content)
    comment_match = re.search(comment_pattern, content, re.MULTILINE)
    
    if response_print_match and comment_match:
        # If we found both, insert between them
        return response_print_match.end() + 1
    elif response_print_match:
        # If only found print statement
        return response_print_match.end() + 1
    else:
        # Default to end of file
        return len(content)


def modify_file(file_path: str, dry_run: bool = False) -> Tuple[str, str]:
    """
    Modify a single API file to save response as CSV
    
    Args:
        file_path: Path to the API file
        dry_run: If True, don't write changes, just return modified content
        
    Returns:
        Tuple of (original content, modified content)
    """
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Make a copy of the original content
    original_content = content
    
    # Add imports if needed
    content = add_imports(content)
    
    # Generate the CSV save code
    csv_save_code = generate_csv_save_code(file_path)
    
    # Find where to insert the code
    insertion_point = find_insertion_point(content)
    
    # Insert the code
    modified_content = content[:insertion_point] + csv_save_code + content[insertion_point:]
    
    # Write the modified file if not a dry run
    if not dry_run:
        with open(file_path, 'w') as f:
            f.write(modified_content)
    
    return original_content, modified_content


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Modify Coinglass API files to save responses as CSV')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t write changes, just print them')
    parser.add_argument('--backup', action='store_true', help='Create backup files (.bak) before modifying')
    parser.add_argument('--single', type=str, help='Modify only a single file (specify path)')
    
    args = parser.parse_args()
    
    if args.single:
        if not os.path.exists(args.single):
            print(f"Error: File not found: {args.single}")
            return
        
        files_to_modify = [args.single]
    else:
        files_to_modify = find_api_files()
    
    print(f"Found {len(files_to_modify)} API files to modify")
    
    for file_path in files_to_modify:
        print(f"Processing: {file_path}")
        
        # Create backup if requested
        if args.backup:
            backup_path = f"{file_path}.bak"
            with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            print(f"  Created backup: {backup_path}")
        
        # Modify the file
        original, modified = modify_file(file_path, args.dry_run)
        
        if args.dry_run:
            print(f"  Dry run - changes not written")
        else:
            print(f"  Modified successfully")


if __name__ == "__main__":
    main()