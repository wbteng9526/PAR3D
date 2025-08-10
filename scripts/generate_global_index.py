#!/usr/bin/env python3
"""
Script to generate a text file containing all subfolders of a given root directory.
Each line in the output file will contain the relative path from the root directory.
"""

import os
import argparse
from pathlib import Path


def get_all_subfolders(root_dir):
    """
    Get all leaf subfolders of the given root directory.
    Only includes directories that have no subdirectories.
    
    Args:
        root_dir (str): Path to the root directory
        
    Returns:
        list: List of relative paths to leaf subfolders
    """
    subfolders = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise ValueError(f"Root directory '{root_dir}' does not exist")
    
    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}' is not a directory")
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Convert to Path object for easier manipulation
        current_path = Path(dirpath)
        
        # Get relative path from root
        relative_path = current_path.relative_to(root_path)
        
        # Skip the root directory itself
        if relative_path == Path('.'):
            continue
            
        # Only include directories that have no subdirectories (leaf directories)
        if not dirnames:
            subfolders.append(str(relative_path))
    
    return sorted(subfolders)


def write_subfolders_to_file(subfolders, output_file):
    """
    Write the list of subfolders to a text file.
    
    Args:
        subfolders (list): List of subfolder paths
        output_file (str): Path to the output file
    """
    with open(output_file, 'w') as f:
        for subfolder in subfolders:
            f.write(f"{subfolder}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a text file containing all subfolders of a given root directory"
    )
    parser.add_argument(
        "root_dir",
        help="Path to the root directory"
    )
    parser.add_argument(
        "-o", "--output",
        default="subfolders.txt",
        help="Output file path (default: subfolders.txt)"
    )
    
    args = parser.parse_args()
    
    try:
        # Get all subfolders
        subfolders = get_all_subfolders(args.root_dir)
        
        # Write to file
        write_subfolders_to_file(subfolders, args.output)
        
        print(f"Found {len(subfolders)} subfolders")
        print(f"Output written to: {args.output}")
        
        # Print first few entries as preview
        if subfolders:
            print("\nFirst few entries:")
            for subfolder in subfolders[:5]:
                print(f"  {subfolder}")
            if len(subfolders) > 5:
                print(f"  ... and {len(subfolders) - 5} more")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
