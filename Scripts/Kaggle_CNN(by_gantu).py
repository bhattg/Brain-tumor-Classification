from keras.models import *
from keras.layers import *
from keras.callbacks import *
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
from sklearn.metrics import*
import keras.activations
from keras.callbacks import EarlyStopping

epsilon= 0.000001
learning_rate=0.001
num_epochs=50
validation_ratio=0.3


root="../input/mid-brain-images/data/data/"
x=[]
s=[]
for i in range(1, 7):
    x_temp = np.load(root+'x_train'+str(i)+'.npy')
    s_temp = np.load(root+'s_train'+str(i)+'.npy')
    x.append(x_temp); s.append(s_temp)

for i in range(1, 6):
    x[0]= np.concatenate((x[0], x[i]))
for i in range(1, 6):
    s[0]= np.concatenate((s[0], s[i]))

x_train, x_val, s_train, s_val = train_test_split(x[0],s[0], test_size=0.3, random_state=42)
mean= np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train-mean)/(std+epsilon)
x_val = (x_val- mean)/(std+epsilon)

x_train = np.expand_dims(x_train, axis=3)
x_val = np.expand_dims(x_val, axis=3)
s_train= np.expand_dims(s_train, axis=1)
s_val= np.expand_dims(s_val, axis=1)

#applying batch norm
def CNN2_model(input_shape):
    X_input = Input(input_shape)  # 180, 180, 1
    out= Conv2D(4, (9, 9),strides= (2, 2))(X_input) #86,86,4
    out= MaxPool2D(strides=(1, 1))(out) #85,85,4
    out= Activation('relu')(out) #85, 85, 4
    out= Conv2D(16, (7, 7),strides=  (2, 2))(out) #40, 40, 16
    out= MaxPool2D(strides=(1, 1))(out) #39, 39, 16
    out= Activation('relu')(out) 
    out= Conv2D(32, (5, 5), strides= (2, 2))(out) #18,18 32
    out= MaxPool2D(strides=(1, 1))(out) #17, 17, 32
    out= Activation('relu')(out) 
    out= Conv2D(64, (5, 5), activation='relu', strides=(2, 2))(out) #7, 7, 64
    out= Flatten()(out)
    out= Dense(2048, activation='relu')(out)
    out= BatchNormalization()(out)
    out= Dense(1024, activation='relu')(out)
    out= BatchNormalization()(out)
    out= Dense(512, activation='relu')(out)
    out= BatchNormalization()(out)
    out= Dense(64, activation='relu')(out)
    out= BatchNormalization()(out)
    out= Dense(10, activation='relu')(out)
    out= Dense(1, activation='sigmoid')(out)
    
    return Model(inputs= X_input, outputs= out)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10,baseline=0.974)
model3= CNN2_model((180, 180, 1))
model3.compile(optimizer='Nadam',loss='binary_crossentropy',metrics=['accuracy'])
history=model3.fit(x=x_train, y=s_train, batch_size=64,callbacks=[es] ,epochs=55,verbose=True, validation_data=(x_val, s_val))
model3.save('saved_model3.h5')
predictions =model3.predict(x_val)
predictions= predictions>=0.5
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_curve
accuracy_score = accuracy_score(s_val, predictions)
recall_score= recall_score(s_val, predictions)
f1_score=f1_score(s_val, predictions)
from scipy.spatial.distance import dice
dice_diss_coeff = dice(s_val, predictions)
DSC= 1-dice_diss_coeff
print("DSC = "+str(DSC))
print("accuracy = "+str(accuracy_score))
print('recall = '+str(recall_score))
print('f1 score = '+str(f1_score))


from scipy.spatial.distance import dice
dice_diss_coeff = dice(s_val, predictions)
DSC= 1-dice_diss_coeff
print(DSC)


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

