import h5py
import os

# Check the CI4R data file
data_file = '/Users/parthbhalodiya/Downloads/C781/AIRHAR/datasets/CI4R/ci4r_data.h5'

if os.path.exists(data_file):
    print(f"Data file exists: {data_file}")
    print(f"File size: {os.path.getsize(data_file)} bytes")

    try:
        with h5py.File(data_file, 'r') as f:
            print("\nKeys in file:", list(f.keys()))

            if 'train' in f:
                train_group = f['train']
                print(f"\nTrain group keys: {list(train_group.keys())}")
                if 'spectrograms' in train_group:
                    print(f"Train spectrograms shape: {train_group['spectrograms'].shape}")
                if 'labels' in train_group:
                    print(f"Train labels shape: {train_group['labels'].shape}")

            if 'test' in f:
                test_group = f['test']
                print(f"\nTest group keys: {list(test_group.keys())}")
                if 'spectrograms' in test_group:
                    print(f"Test spectrograms shape: {test_group['spectrograms'].shape}")
                if 'labels' in test_group:
                    print(f"Test labels shape: {test_group['labels'].shape}")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"Data file does not exist: {data_file}")

# Check raw data
raw_data_path = '/Users/parthbhalodiya/Downloads/C781/AIRHAR/datasets/CI4R/Cross-frequency/'
if os.path.exists(raw_data_path):
    print(f"\n\nRaw data directory exists: {raw_data_path}")
    subdirs = [d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))]
    print(f"Subdirectories: {subdirs}")

    for subdir in subdirs[:3]:  # Check first 3 subdirs
        subdir_path = os.path.join(raw_data_path, subdir)
        files = os.listdir(subdir_path)
        bin_files = [f for f in files if f.endswith('.bin')]
        print(f"\n{subdir}: {len(bin_files)} .bin files")
