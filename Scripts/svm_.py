import numpy as np 
from sklearn import svm

Path_X_train = ""
Path_Y_train = ""
Path_X_val= ""
Path_Y_val = ""

x_train = np.load(Path_X_train)
y_train = np.load(Path_Y_train)
x_val = np.load(Path_X_val)
y_val = np.load(Path_Y_val)

clf= svm.SVC(class_weight='balanced')

clf.fit(x_train, y_train)
print("Train accuracy : "+str(clf.score(x_train, y_train)))
print('Validation accuracy : '+str(clf.score(x_val, y_val)))

