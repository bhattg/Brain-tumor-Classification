import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt


#paths need to be revamped and checked before doing anything

X= np.load('C:\\Users\\asus\\Desktop\\DL testing\\data\\Validation_x.npy')
y= np.load('C:\\Users\\asus\\Desktop\\DL testing\\data\\Validation_y.npy')

loc=444
plt.figure()
plt.imshow(X[loc,:,:])
x = ss.medfilt2d(X[loc,:,:])

plt.figure()
plt.imshow(x)

diff= X[loc,:,:]-x

plt.figure()
plt.imshow(diff, cmap="Greys")
