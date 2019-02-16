import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import math


###############Auto Encoder

EPSILON = 1e-10
batches = 128
num_epochs = 100
learning_rate = 1

X_train = np.load(os.path.join("Data", "Train_x.npy"))
Y_train = np.load(os.path.join("Data", "Train_y.npy"))
X_val = np.load(os.path.join("Data", "Validation_x.npy"))
Y_val = np.load(os.path.join("Data", "Validation_x.npy"))

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / (std + EPSILON)
X_val = (X_val - mean) / (std + EPSILON)

X_train = X_train[:, 40:220, 30:210]
Y_train = Y_train[:, 40:220, 30:210]
X_val = X_val[:, 40:220, 30:210]
Y_val = Y_val[:, 40:220, 30:210]

X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)
Y_train = np.expand_dims(Y_train, axis=1)
Y_val = np.expand_dims(Y_val, axis=1)

Y_train = Y_train.astype(int)
Y_val = Y_val.astype(int)
############## conversion to tensor
tensor_x = torch.Tensor(X_train)
tensor_y = torch.Tensor(Y_train)
tensor_x_val = torch.Tensor(X_val)
tensor_y_val = torch.Tensor(Y_val)

train_set = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
val_set = utils.TensorDataset(tensor_x_val, tensor_y_val)

##############loader stream
train_loader = utils.DataLoader(
    dataset=train_set, batch_size=batches, shuffle=True)
val_loader = utils.DataLoader(
    dataset=val_set, batch_size=batches, shuffle=True)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
print(device)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.ReLU(),
            nn.Conv2d(4, 16, 3), nn.MaxPool2d(2, return_indices=True), nn.ReLU(),
            nn.Conv2d(16, 64, 3),nn.ReLU(),
            nn.Conv2d(64, 128, 3), nn.MaxPool2d(2,return_indices=True), nn.ReLU(),
        )
        self.Decoder = nn.Sequential(
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=3, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels= 16,kernel_size=3, stride=2), nn.ReLU(),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=3, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=2)     
        )
        
        self._initialize_submodules()

    def forward(self, x):
        out = self.Encoder(x)
        out = self.Decoder(out)
        return out

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))


model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
for epoch in range(num_epochs):
    acc = 0
    count = 0
    total_num_batches = len(train_loader)
    for image, labels in tqdm(train_loader):
        image = image.to(device)
        labels = labels.to(device)
        output = model(image).view(-1)
        optimizer.zero_grad()
        loss = F.mse_loss(output, labels)
        loss.backward()
        optimizer.step()

    #     output = (output >= 0.5).float()
    #     acc += (output == labels).sum().item()
    #     count += labels.shape[0]

    # loss = loss / total_num_batches
    # acc /= count

    print("Training loss =", loss.item())
    # print("Training acc =", acc)

    if(epoch % 5 == 0):
        acc = 0
        count = 0
        total_num_batches = len(train_loader)
        for image, labels in val_loader:
            image = image.to(device)
            labels = labels.to(device)
            output = model(image).view(-1)
            loss = F.mse_loss(output, labels)

        #     output = (output >= 0.5).float()
        #     acc += (output == labels).sum().item()
        #     count += labels.shape[0]

        # loss = loss / total_num_batches
        # acc /= count

        print("validation loss =", loss.item())
        print()


os.makedirs("Models", exist_ok=True)
torch.save(model.state_dict(), os.path.join("Models"))
