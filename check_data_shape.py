import h5py

data_path = "datasets/CI4R/ci4r_data.h5"

with h5py.File(data_path, 'r') as f:
    X_test = f['test/spectrograms'][:]
    y_test = f['test/labels'][:]

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_test dtype: {X_test.dtype}")
    print(f"First sample shape: {X_test[0].shape}")
    print(f"First sample min/max: {X_test[0].min():.3f} / {X_test[0].max():.3f}")
