import numpy as np
import sys
from pathlib import Path

def print_npz_keys(npz_file, prefix=''):
    """
    Recursively print all keys in an .npz file, handling nested dictionaries.
    
    Args:
        npz_file: Path to the .npz file or loaded NpzFile object
        prefix: String prefix for nested key indentation
    """
    # Load .npz file if input is a file path
    if isinstance(npz_file, (str, Path)):
        try:
            npz = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading .npz file: {e}")
            return
    else:
        npz = npz_file

    # Iterate through all keys in the .npz file
    for key in npz.files:
        print(f"{prefix}{key}")
        # Check if the item is a dictionary or an object that might contain nested data
        item = npz[key]
        if isinstance(item, np.ndarray) and item.dtype == object:
            try:
                # If the item is a dictionary, recurse into it
                if isinstance(item.item(), dict):
                    print_dict_keys(item.item(), prefix + '  ')
            except:
                pass
        elif isinstance(item, dict):
            # Directly handle dictionary
            print_dict_keys(item, prefix + '  ')

def print_dict_keys(d, prefix=''):
    """
    Recursively print keys in a dictionary.
    
    Args:
        d: Dictionary to print
        prefix: String prefix for nested key indentation
    """
    for key, value in d.items():
        print(f"{prefix}{key}")
        if isinstance(value, dict):
            print_dict_keys(value, prefix + '  ')

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_npz_keys.py <path_to_npz_file>")
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    if not npz_path.exists():
        print(f"File not found: {npz_path}")
        sys.exit(1)

    print(f"Keys in {npz_path}:")
    print_npz_keys(npz_path)

if __name__ == "__main__":
    main()