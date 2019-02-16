from keras.models import *
from keras.layers import *
from keras.callbacks import  EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

x_all =  np.load('../input/assignment-1/data_pca/x_net_pca.npy')
y_all=   np.load('../input/assignment-1/data_pca/s_net_pca.npy')
def nn(input_shape):
    X_input= Input(input_shape)
    out= Dense(5, activation='relu')(X_input)
    out= Dense(2, activation='relu')(out)
    out= Dense(1, activation='sigmoid')(out)

    return Model(inputs= X_input, outputs=out)
x_train, x_validate, y_train, y_validate = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
nn_model = nn((10,))
nn_model.compile(optimizer='Nadam',loss='binary_crossentropy',metrics=['accuracy'])
nn_model.fit(x=x_train, y=y_train, batch_size=16, epochs=50,verbose=True, validation_data=(x_validate,y_validate))