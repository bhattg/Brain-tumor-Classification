from PIL import Image 
import numpy as np
import os
from tqdm import tqdm
root='C:\\Users\\asus\Downloads\\normalsVsAbnormalsV1\\abnormalsJPG\\'
root_2='C:\\Users\\asus\Downloads\\normalsVsAbnormalsV1\\normalsJPG\\'
x_abnormal_list=[]
x_normal = []
s_abnormal_list=[]
s_normal=[]
for filename in tqdm(os.listdir(root)):
    x_jpg =Image.open(root+filename)
    x_jpg = x_jpg.convert('L')
    x_np = np.asarray(x_jpg)
    x_np = x_np[20:200, 20:200]
    x_abnormal_list.append(x_np)
    s_abnormal_list.append(1)
for filename in tqdm(os.listdir(root_2)):
    x_jpg =Image.open(root_2+filename)
    x_jpg = x_jpg.convert('L')
    x_np = np.asarray(x_jpg)
    x_np = x_np[20:200, 20:200]
    x_normal.append(x_np)
    s_normal.append(0)

np.save('x_normal.npy', np.asarray(x_normal))
print('x_normal saved!')
np.save('s_normal.npy', np.asarray(s_normal))
print('x_normal saved!')
np.save('x_abnormal.npy', np.asarray(x_abnormal_list))
print('x_normal saved!')
np.save('s_abnormal.npy', np.asarray(s_abnormal_list))
print('x_normal saved!')