#!/usr/bin/env python3
"""
CoinGlass API Data Collection Tool

This script allows you to select and run specific API endpoints from the coinglass-api folder
and save the data to matching locations in the data/coinglass folder.
"""

import os
import subprocess
import sys
from typing import List, Dict, Tuple
import argparse


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
    
    return sorted(api_files)


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
    file_indices = {}  # Maps indices to file paths
    
    # Sort categories for consistent display
    for category in sorted(categories.keys()):
        files = sorted(categories[category])
        print(f"\n--- {category} ---")
        for file_path in files:
            filename = os.path.basename(file_path)
            print(f"{file_index}. {filename}")
            file_indices[file_index] = file_path
            file_index += 1
    
    print("\nOptions:")
    print("A. Run all API endpoints")
    print("C. Run by category")
    print("Q. Quit")
    
    return file_indices


def run_api_file(file_path: str) -> bool:
    """Run a single API file and return success status."""
    print(f"Running: {file_path}")
    try:
        subprocess.run([sys.executable, file_path], check=True)
        print(f"✓ Completed: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running: {file_path}")
        print(f"  Error details: {e}")
        return False
    

def run_selected_files(file_indices: Dict[int, str], selections: List[int]) -> None:
    """Run the selected API files."""
    total = len(selections)
    successful = 0
    
    for idx in selections:
        if idx in file_indices:
            if run_api_file(file_indices[idx]):
                successful += 1
    
    print(f"\nCompleted {successful} of {total} API requests.")


def get_categories(api_files: List[Tuple[str, str]]) -> Dict[str, List[int]]:
    """Get a dictionary mapping categories to file indices."""
    categories = {}
    
    for i, (file_path, category) in enumerate(api_files, 1):
        if category not in categories:
            categories[category] = []
        categories[category].append(i)
    
    return categories


def run_by_category(api_files: List[Tuple[str, str]], file_indices: Dict[int, str]) -> None:
    """Allow user to run all files in a specific category."""
    categories = get_categories(api_files)
    
    print("\nAvailable categories:")
    for i, category in enumerate(sorted(categories.keys()), 1):
        print(f"{i}. {category} ({len(categories[category])} endpoints)")
    
    try:
        choice = input("\nSelect category (number): ").strip()
        cat_idx = int(choice)
        
        category_name = sorted(categories.keys())[cat_idx-1]
        print(f"\nRunning all endpoints in category: {category_name}")
        
        run_selected_files(file_indices, categories[category_name])
        
    except (ValueError, IndexError):
        print("Invalid category selection.")


def main():
    parser = argparse.ArgumentParser(description="CoinGlass API Data Collection Tool")
    parser.add_argument(
        "--run-all", 
        action="store_true", 
        help="Run all API endpoints without interactive menu"
    )
    parser.add_argument(
        "--category", 
        type=str, 
        help="Run all endpoints in specific category"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available API endpoints and categories"
    )
    
    args = parser.parse_args()
    api_files = get_all_api_files()
    
    # Display list mode
    if args.list:
        file_indices = display_menu(api_files)
        return
    
    # Non-interactive mode
    if args.run_all:
        print("Running all API endpoints...")
        file_indices = {i: file_path for i, (file_path, _) in enumerate(api_files, 1)}
        run_selected_files(file_indices, range(1, len(api_files) + 1))
        return
        
    if args.category:
        categories = get_categories(api_files)
        category_match = None
        
        # Try to match the category
        for category in categories:
            if args.category.lower() in category.lower():
                category_match = category
                break
                
        if category_match:
            print(f"Running all endpoints in category: {category_match}")
            file_indices = {i: file_path for i, (file_path, _) in enumerate(api_files, 1)}
            run_selected_files(file_indices, categories[category_match])
        else:
            print(f"Category '{args.category}' not found. Available categories:")
            for category in sorted(categories.keys()):
                print(f"- {category}")
        return
    
    # Interactive mode
    while True:
        file_indices = display_menu(api_files)
        
        choice = input("\nEnter selection (comma-separated numbers, 'A' for all, 'C' for category, or 'Q' to quit): ").strip().upper()
        
        if choice == 'Q':
            print("Exiting...")
            break
        
        elif choice == 'A':
            print("\nRunning all API endpoints...")
            run_selected_files(file_indices, list(file_indices.keys()))
        
        elif choice == 'C':
            run_by_category(api_files, file_indices)
        
        else:
            try:
                # Parse comma-separated selections
                selections = [int(x.strip()) for x in choice.split(',')]
                run_selected_files(file_indices, selections)
            except ValueError:
                print("Invalid selection. Please enter numbers separated by commas, 'A', 'C', or 'Q'.")


if __name__ == "__main__":
    main()