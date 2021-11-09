from tqdm import tqdm
import numpy as np
import argparse
import os
import random

# pytorch
import torch
from models import QVRNN
from torch.utils.data import Dataset, DataLoader

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store', default=None,
                    dest='datas',
                    help='choose sequence file')
parser.add_argument('-dq', action='store', default=None,
                    dest='dataq',
                    help='choose quality value file')
parser.add_argument('-rate', action='store', default=None,
                    dest='sample_rate',
                    help='sample rate')
parser.add_argument('-data_params', action='store', 
                    dest='params_file',
                    help='params file')
parser.add_argument('-model', action='store',
                    dest='model_file_temp',
                    help='weights will be stored with this name')
parser.add_argument('-log_file', action='store',
                    dest='log_file',
                    help='Log file')
arguments = parser.parse_args()
print(arguments)

class myDataset(Dataset):
        def __init__(self, file_path, q_file_path, time_steps, sample_rate):
                series = np.load(file_path)['data']
                datas = self.strided_app(series, time_steps+1, 1)

                self.x = datas[:, :-1]
                self.y = datas[:, -1]

                series_q = np.load(q_file_path)['data']
                data_q = self.strided_app(series_q, time_steps+1, 1)
                self.q = data_q[:, :-1]
                row_len = datas.shape[0]
                total_idx = list(range(row_len))
                sample_idx = random.sample(total_idx, int(row_len * sample_rate))
                sample_idx.sort()
                self.x = self.x[sample_idx]
                self.y = self.y[sample_idx]
                self.q = self.q[sample_idx]
                # np.int64, int64, float64

        def __len__(self):
                return len(self.x)
 
        def __getitem__(self, idx):
                inputx = torch.tensor(self.x[idx]).type(torch.LongTensor)
                inputq = torch.tensor(self.q[idx]).type(torch.LongTensor)
                label = torch.tensor(self.y[idx]).type(torch.LongTensor)
                # (64,2) (1)
                return inputx, inputq, label

        # 滑动窗口切分数组
        def strided_app(self, a, L, S):  # Window len = L, Stride len/stepsize = S
                nrows = ((a.size - L) // S) + 1
                n = a.strides[0]
                return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

batch_size=128
sequence_length=64
num_epochs=5
num_gpus=torch.cuda.device_count()
sample_rate=float(arguments.sample_rate)

with open(arguments.params_file, 'r') as f:
        alphabet_size = int(f.readline())

# pytorch
train_data = myDataset(arguments.datas, arguments.dataq, sequence_length, sample_rate)
trainloader = DataLoader(train_data, batch_size*num_gpus, shuffle=True, num_workers=4, pin_memory=True)
model = QVRNN(alphabet_size)
model = torch.nn.DataParallel(model)
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
        for inputxs, inputqs, labels in tqdm(trainloader):
                inputxs = inputxs.cuda()
                inputqs = inputqs.cuda()
                labels = labels.cuda()
                # (1) 初始化梯度
                optimizer.zero_grad() 
                # (2) 前向传播
                outputs = model(inputxs, inputqs)
                # (128,4)float32, (128)int64
                loss = criterion(outputs, labels)
                # (3) 反向传播
                loss.backward()
                # (4) 计算损失并更新权重
                optimizer.step()

torch.save(model.module.state_dict(), arguments.model_file_temp)
