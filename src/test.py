import h5py
import numpy as np

file_path = 'data/1DHeatEq/heat1d_data.h5'
with h5py.File(file_path, 'r') as f:
    # Print all top-level keys (like top-level folders)
    def print_h5_structure(name, obj):
        print(name)

    print("File structure:")
    f.visititems(print_h5_structure)

    # If you know the data path, you can access specific datasets
    if 'data' in f:
        data = f['data'][:]  # Read data content as NumPy array
        print("Data content:", data)