from keras.models import *
import keras
import numpy as np
from sklearn.utils import shuffle
import h5py
#原验证集的准确率测试
val_bottle_path='/home/ross/Documents/AIC/Inception_resv2_test_big.h5'
model_path='/home/ross/Documents/AIC/My_trained_multiple1.h5'

X_valid = []
for filename in ['Inception_resv2_val_big.h5']:
    with h5py.File(filename, 'r') as h:
    	X_valid.append(np.array(h['test']))#特征量
    	y_valid = np.array(h['valid'])#标签
X_valid = np.concatenate(X_valid, axis=1)
y_valid = keras.utils.to_categorical(y_valid, 80)
X_valid, y_valid = shuffle(X_valid, y_valid)

model=load_model(model_path)
y_pred = model.predict(X_valid)

accuracy1=0
accuracy2=0

m=y_pred.shape[0]
for i in range(m):
    y=y_pred[i]
    fact=np.argmax(y_valid[i])
    num=3
    label=[]
    while num>0:
        label.append(int(np.argmax(y)))
        y[np.argmax(y)]=0
        num-=1
    if label[0]==fact:
        accuracy1+=1
    if fact in label:
        accuracy2+=1
print("一对一的准确率是：",accuracy1/m)
print("三对一的准确率是：",accuracy2/m)