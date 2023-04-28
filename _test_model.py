import os
import torch
import numpy as np

num_list = []
for file in os.listdir('labeled_data'):
    file_path = os.path.join('labeled_data', file + '/tmp.tar')
    data = torch.load(file_path)
    print(file, len(data[0]))
    num_list.append(len(data[0]))
num_list = np.array(num_list)
print(num_list.min(), num_list.mean(), num_list.max())
print(sum(num_list < 5000))
