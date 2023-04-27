import os
import torch

for file in os.listdir('labeled_data'):
    file_path = os.path.join('labeled_data', file + '/tmp.tar')
    data = torch.load(file_path)
    print(file, len(data[0]))
