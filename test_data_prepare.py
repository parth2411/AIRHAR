import os
import sys
sys.path.insert(0, '/Users/parthbhalodiya/Downloads/C781/AIRHAR')

# Check what the script is looking for
base_dir = "datasets/CI4R/Cross-frequency/"
print(f"Checking path: {base_dir}")
print(f"Path exists: {os.path.exists(base_dir)}")
print(f"Is directory: {os.path.isdir(base_dir)}")

if os.path.exists(base_dir):
    print(f"\nContents: {os.listdir(base_dir)}")

    # Check for directories
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"\nDirectories: {dirs}")

    # Test get_identifiers function
    identifiers = list(set(name.split('_')[0] for name in os.listdir(base_dir)))
    print(f"\nIdentifiers found: {identifiers}")

    # Check a specific activity folder
    test_dir = "77ghz"
    test_path = os.path.join(base_dir, test_dir)
    if os.path.exists(test_path):
        subdirs = os.listdir(test_path)
        print(f"\nSubdirectories in {test_dir}: {subdirs[:5]}")

        # Check files in first activity
        if subdirs:
            activity_path = os.path.join(test_path, subdirs[0])
            if os.path.isdir(activity_path):
                files = os.listdir(activity_path)
                bin_files = [f for f in files if f.endswith('.bin')]
                print(f"\n.bin files in {subdirs[0]}: {len(bin_files)}")
                if bin_files:
                    print(f"Example file: {bin_files[0]}")
