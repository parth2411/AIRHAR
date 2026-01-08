import os
import numpy as np
import h5py
import scipy.io
from typing import List
import cv2
import argparse

def load_mat_file(filepath):
    """
    Load .mat file and extract the radar spectrogram data
    """
    try:
        mat_data = scipy.io.loadmat(filepath)

        # CI4R uses 'sx1' key for spectrogram data
        if 'sx1' in mat_data:
            data = mat_data['sx1']
            return data

        # Fallback: try common keys
        possible_keys = ['data', 'matrix', 'Raw_0', 'spectrogram', 'image', 'sx2']
        for key in possible_keys:
            if key in mat_data:
                data = mat_data[key]
                return data

        # If no common key found, use first non-metadata key
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if keys:
            data = mat_data[keys[0]]
            return data
        else:
            print(f"Warning: No data keys found in {filepath}")
            return None

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_spectrogram(data, target_size=(224, 224)):
    """
    Process the loaded complex spectrogram data into normalized format

    Similar to RGB_mD function in data_prepare_CI4R.py
    """
    # If data is complex, convert to magnitude in dB scale
    if np.iscomplexobj(data):
        # Calculate magnitude
        magnitude = np.abs(data)
        # Convert to dB scale (clipped to avoid log(0))
        data = 20 * np.log10(magnitude + 1e-10)
        # Clip to reasonable range
        data = np.clip(data, -45, data.max())

    # Convert to float32
    data = data.astype(np.float32)

    # Resize to target size if needed
    if data.shape != target_size:
        data = cv2.resize(data, target_size, interpolation=cv2.INTER_CUBIC)

    # Normalize (per-sample normalization)
    data = (data - data.mean()) / (data.std() + 1e-8)

    return data

def one_hot_encode(index: int, num_classes: int) -> np.ndarray:
    """Create a one-hot encoded vector."""
    encoding = np.zeros(num_classes)
    encoding[index] = 1
    return encoding

def prepare_ci4r_from_mat(base_path, train_split=0.8):
    """
    Prepare CI4R dataset from .mat files

    Args:
        base_path: Path to activity_spect_matrices directory
        train_split: Ratio for training data (rest goes to test)
    """
    # Get all activity directories
    activity_dirs = sorted([d for d in os.listdir(base_path)
                           if os.path.isdir(os.path.join(base_path, d))])

    print(f"Found {len(activity_dirs)} activity classes:")
    for i, activity in enumerate(activity_dirs):
        print(f"  Class {i}: {activity}")

    num_classes = len(activity_dirs)

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for class_idx, activity_dir in enumerate(activity_dirs):
        print(f"\nProcessing class {class_idx}: {activity_dir}")
        dir_path = os.path.join(base_path, activity_dir)

        # Get all .mat files
        mat_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.mat')])
        print(f"  Found {len(mat_files)} .mat files")

        if len(mat_files) == 0:
            print(f"  Warning: No .mat files found in {activity_dir}")
            continue

        spectrograms = []
        labels = []

        for mat_file in mat_files:
            file_path = os.path.join(dir_path, mat_file)

            # Load and process the .mat file
            data = load_mat_file(file_path)

            if data is None:
                continue

            # Process into spectrogram
            spectrogram = process_spectrogram(data)
            spectrograms.append(spectrogram)
            labels.append(one_hot_encode(class_idx, num_classes))

        if len(spectrograms) == 0:
            print(f"  Warning: No valid spectrograms extracted from {activity_dir}")
            continue

        spectrograms = np.array(spectrograms)
        labels = np.array(labels)

        print(f"  Extracted {len(spectrograms)} spectrograms")

        # Split into train/test
        # Use last 12 samples for testing (consistent with original script)
        train_size = max(1, len(spectrograms) - 12)

        X_train_class = spectrograms[:train_size]
        X_test_class = spectrograms[train_size:]
        y_train_class = labels[:train_size]
        y_test_class = labels[train_size:]

        print(f"  Train: {len(X_train_class)}, Test: {len(X_test_class)}")

        X_train_list.append(X_train_class)
        X_test_list.append(X_test_class)
        y_train_list.append(y_train_class)
        y_test_list.append(y_test_class)

    # Concatenate all classes
    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print(f"\n=== Final Dataset Statistics ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {num_classes}")
    print(f"Spectrogram shape: {X_train[0].shape}")

    # Save to h5py format
    output_dir = "datasets/CI4R"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ci4r_data.h5')

    with h5py.File(output_path, 'w') as f:
        # Create train group
        train_group = f.create_group('train')
        train_group.create_dataset('spectrograms', data=X_train, dtype='float32')
        train_group.create_dataset('labels', data=y_train, dtype='float32')

        # Create test group
        test_group = f.create_group('test')
        test_group.create_dataset('spectrograms', data=X_test, dtype='float32')
        test_group.create_dataset('labels', data=y_test, dtype='float32')

    print(f"\nDataset saved to {output_path}")
    return num_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare CI4R dataset from .mat files')
    parser.add_argument('--path', type=str,
                       default='/Users/parthbhalodiya/Downloads/C781/AIRHAR/datasets/CI4R/Cross-frequency/77ghz/activity_spect_matrices',
                       help='Path to activity_spect_matrices directory')
    args = parser.parse_args()

    num_classes = prepare_ci4r_from_mat(args.path)
    print(f"\n✓ Successfully processed CI4R dataset with {num_classes} classes")
