import matplotlib.pyplot as plt
import torch
import os
from scipy import io as sio
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
x = torch.tensor(sio.loadmat("x.mat")['x'])
numpy_array_2d = x[0][0].numpy()
plt.imshow(numpy_array_2d, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.show()
