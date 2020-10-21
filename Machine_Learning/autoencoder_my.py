import torch
import torch.nn as nn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# 超参数
EPOCH = 200
LR = 0.005



data = load_iris()
y = data.target
x = data.data


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()




for epoch in range(EPOCH):
     b_x = torch.from_numpy(x).unsqueeze(0).float()
     b_y = torch.from_numpy(x).unsqueeze(0).float()
     _, decoded = autoencoder(b_x)
     loss = loss_func(decoded, b_y)
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()

 #    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())


encoded_data, _ = autoencoder(torch.from_numpy(x).unsqueeze(0).float())
x = encoded_data.detach().numpy().squeeze(0)
print(encoded_data.detach().numpy().shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(y_train)
print(type(y_train))

# class RNN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.rnn=torch.nn.LSTM(
#             input_size=2,
#             hidden_size=64,
#             num_layers=1,
#             batch_first=True
#         )
#         self.out = torch.nn.Linear(in_features=64, out_features=3)

#     def forward(self, x):
#         # 一下关于shape的注释只针对单项
#         # output: [batch_size, time_step, hidden_size]
#         # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
#         # c_n: 同h_n
#         output, (h_n, c_n) = self.rnn(x)
#         # output_in_last_timestep=output[:,-1,:] # 也是可以的
#         output_in_last_timestep = h_n[-1, :, :]
#         # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
#         x = self.out(output_in_last_timestep)
#         return x


# net = RNN()
# # 3. 训练
# # 3. 网络的训练（和之前CNN训练的代码基本一样）
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# loss_F = torch.nn.CrossEntropyLoss()
# for epoch in range(500):  # 数据集只迭代一次

#     x = torch.from_numpy(X_train).unsqueeze(0).float()
#     y = torch.from_numpy(y_train).long()
#     pred = net(x.view(-1, 1, 2))

#     loss = loss_F(pred, y)  # 计算loss
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())


# with torch.no_grad():
#         test_pred = net(torch.from_numpy(X_test).unsqueeze(0).float().view(-1, 1, 2))
#         Y_test = torch.from_numpy(y_test).long()
#         prob = torch.nn.functional.softmax(test_pred, dim=1)
#         pred_cls = torch.argmax(prob, dim=1)
#         acc = (pred_cls == Y_test).sum().numpy() / pred_cls.size()[0]
#         print(f"{epoch}: accuracy:{acc}")