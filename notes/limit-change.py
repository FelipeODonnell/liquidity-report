#!/usr/bin/env python3
"""
Script to add a limit parameter of 4500 to API URLs that have an interval parameter.
"""

import os
import re
from pathlib import Path

def add_limit_to_url(url_string):
    """
    Add a limit=4500 parameter to a URL that has an interval parameter.
    
    Args:
        url_string: The URL string
        
    Returns:
        Modified URL string or original if no changes were made
    """
    # Check if the URL has an interval parameter
    if 'interval=' not in url_string:
        return url_string
    
    # Check if URL already has a limit parameter
    if 'limit=' in url_string:
        # Replace existing limit parameter
        url_string = re.sub(r'limit=\d+', 'limit=4500', url_string)
    else:
        # Add limit parameter
        if '?' in url_string:
            # URL already has parameters, add limit as an additional parameter
            url_string = url_string + '&limit=4500'
        else:
            # URL has no parameters, add limit as the first parameter
            url_string = url_string + '?limit=4500'
    
    return url_string

def process_file(file_path):
    """
    Process a single file to add limit parameter to URLs with interval.
    
    Args:
        file_path: Path to the Python file to process
        
    Returns:
        (bool) True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find URL assignments in the content
        url_pattern = r'(url\s*=\s*")(.*?)(")' 
        
        def replace_url(match):
            prefix = match.group(1)
            url = match.group(2)
            suffix = match.group(3)
            
            if 'interval=' in url:
                new_url = add_limit_to_url(url)
                return prefix + new_url + suffix
            return match.group(0)
        
        new_content = re.sub(url_pattern, replace_url, content)
        
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    """
    Find all Python files in the coinglass-api directory and add limit parameter
    to URLs that have an interval parameter.
    """
    api_dir = "coinglass-api"
    if not os.path.exists(api_dir):
        print(f"Error: Directory '{api_dir}' not found.")
        return

    modified_files = 0
    total_files = 0
    modified_file_paths = []
    
    # Traverse the directory and process each Python file
    for root, dirs, files in os.walk(api_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                total_files += 1
                
                if process_file(file_path):
                    modified_files += 1
                    modified_file_paths.append(file_path)
    
    print(f"\nSummary:")
    print(f"  - Scanned {total_files} Python files in {api_dir}")
    print(f"  - Modified {modified_files} files to add 'limit=4500' to URLs with 'interval' parameter")
    
    if modified_files > 0:
        print("\nFiles modified:")
        for path in modified_file_paths:
            print(f"  - {path}")

if __name__ == "__main__":
    main()