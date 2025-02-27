from data_h import HISRDatasets
from torch.utils.data import DataLoader
from ipdb import set_trace

train_path = "G:\\PythonProjectGpu\\RRConv\\training_data\\harvard_x4\\test_harvardv3(with_up)x4.h5"
val_path = "G:\\PythonProjectGpu\\RRConv\\training_data\\harvard_x4\\train_harvard(with_up)x4.h5"

train_set = HISRDatasets(file=train_path)
val_set = HISRDatasets(file=val_path)


training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=32, shuffle=True,
                                    pin_memory=True, drop_last=False)
validate_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=32, shuffle=True,
                                    pin_memory=True, drop_last=False)
set_trace()
print("DataLoader created successfully!")
                                