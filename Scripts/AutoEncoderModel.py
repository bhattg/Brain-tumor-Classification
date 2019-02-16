#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch 
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.utils.data as utils
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt


# In[ ]:


tensor_x = torch.Tensor(X_train)
tensor_y = torch.Tensor(Y_train)
tensor_x_val= torch.Tensor(X_val)
tensor_y_val= torch.Tensor(Y_val)

train_set = utils.TensorDataset(tensor_x,tensor_y) # create your datset
val_set= utils.TensorDataset(tensor_x_val,tensor_y_val)

train_loader = utils.DataLoader(dataset = train_set,batch_size=64,shuffle=True)
val_loader= utils.DataLoader(dataset = val_set,batch_size=64,shuffle=True)


# In[4]:


num_epoch=5
learning_rate=0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[6]:


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.Encoder= nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1,  return_indices=True),     
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, return_indices=True)    # at this point dimension (I am writing channel last for my convenience)= 38*38*16        
        )
        self.Decoder= nn.Sequential(
            nn.MaxUnpool2d(kernel_size=3, stride=1),
            nn.ConvTranspose2d(kernel_size=5, stride=2),
            nn.MaxUnpool2d(kernel_size=3, stride=1),
            nn.ConvTranspose2d(kernel_size=5, stride=2),
            nn.ConvTranspose2d(kernel_size=3, stride=1)
        )
        
        def forward(self, x):
            x= self.Encoder(x)
            x= self.Decoder(x)
            return x
    


# In[7]:


model = AutoEncoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


# In[ ]:




