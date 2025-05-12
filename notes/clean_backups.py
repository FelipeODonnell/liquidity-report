#!/usr/bin/env python3
"""
Remove all .bak files from the coinglass-api directory
"""

import os
import glob

def remove_backup_files():
    """Find and remove all .bak files in the coinglass-api directory"""
    root_dir = 'coinglass-api'
    
    # Find all .bak files
    backup_files = glob.glob(f"{root_dir}/**/*.bak", recursive=True)
    
    # Count found files
    total_files = len(backup_files)
    print(f"Found {total_files} backup files to remove")
    
    # Remove each file
    for file_path in backup_files:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    print(f"\nCompleted: Removed {total_files} backup files")

if __name__ == "__main__":
    remove_backup_files()