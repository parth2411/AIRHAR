import os
import numpy as np
import h5py
import cv2
from typing import List

def load_spectrogram_png(filepath):
    """Load and preprocess a spectrogram PNG file"""
    # Read image as grayscale
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to 224x224 if needed
    if img.shape != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img

def one_hot_encode(index: int, num_classes: int) -> np.ndarray:
    """Create a one-hot encoded vector"""
    encoding = np.zeros(num_classes)
    encoding[index] = 1
    return encoding

def prepare_from_spectrograms(base_path, test_samples_per_class=12):
    """
    Prepare dataset from pre-computed spectrogram PNG files

    Args:
        base_path: Path to the activity_spectogram_77GHz directory
        test_samples_per_class: Number of samples to reserve for testing per class
    """
    print(f"Loading spectrograms from: {base_path}")

    # Get all activity directories
    activity_dirs = sorted([d for d in os.listdir(base_path)
                          if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')])

    print(f"Found {len(activity_dirs)} activity classes:")
    for i, activity in enumerate(activity_dirs):
        print(f"  {i}: {activity}")

    num_classes = len(activity_dirs)

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for class_idx, activity_dir in enumerate(activity_dirs):
        activity_path = os.path.join(base_path, activity_dir)

        # Get all PNG files
        png_files = sorted([f for f in os.listdir(activity_path) if f.endswith('.png')])

        print(f"\nProcessing {activity_dir}: {len(png_files)} samples")

        # Load all spectrograms for this class
        spectrograms = []
        for png_file in png_files:
            filepath = os.path.join(activity_path, png_file)
            spec = load_spectrogram_png(filepath)
            if spec is not None:
                spectrograms.append(spec)

        if len(spectrograms) == 0:
            print(f"  WARNING: No valid spectrograms found for {activity_dir}")
            continue

        spectrograms = np.array(spectrograms)
        labels = np.array([one_hot_encode(class_idx, num_classes) for _ in range(len(spectrograms))])

        # Split into train and test
        train_size = max(1, len(spectrograms) - test_samples_per_class)

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

    print(f"\n=== Dataset Summary ===")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Spectrogram shape: {X_train.shape[1:]}")
    print(f"Number of classes: {num_classes}")

    # Normalize per-sample
    print("\nNormalizing spectrograms...")
    X_train = np.array([
        (spec - spec.mean()) / (spec.std() + 1e-8)
        for spec in X_train
    ])
    X_test = np.array([
        (spec - spec.mean()) / (spec.std() + 1e-8)
        for spec in X_test
    ])

    # Save to h5py format
    output_dir = "datasets/CI4R"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ci4r_data.h5')

    print(f"\nSaving to: {output_file}")
    with h5py.File(output_file, 'w') as f:
        # Create train group
        train_group = f.create_group('train')
        train_group.create_dataset('spectrograms', data=X_train, dtype='float32')
        train_group.create_dataset('labels', data=y_train, dtype='float32')

        # Create test group
        test_group = f.create_group('test')
        test_group.create_dataset('spectrograms', data=X_test, dtype='float32')
        test_group.create_dataset('labels', data=y_test, dtype='float32')

    print("Done!")
    return output_file

if __name__ == "__main__":
    base_path = "datasets/CI4R/Cross-frequency/Spectograms_77_24_Xethrue/activity_spectogram_77GHz"
    prepare_from_spectrograms(base_path)
