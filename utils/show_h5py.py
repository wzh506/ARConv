import h5py

def print_hdf5_structure(name, obj):
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}, Shape: {obj.shape}, DataType: {obj.dtype}")
        # 打印数据集的属性
        for attr_key, attr_value in obj.attrs.items():
            print(f"{indent}  Attribute - {attr_key}: {attr_value}")

def view_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as hdf:
        print(f"HDF5 文件: {file_path}")
        hdf.visititems(print_hdf5_structure)

if __name__ == "__main__":
    hdf5_file = 'G:\\PythonProjectGpu\\RRConv\\training_data\\train_wv3.h5'  # 替换为你的HDF5文件路径
    view_hdf5_structure(hdf5_file)

