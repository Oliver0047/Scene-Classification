import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread,imresize 
from keras.models import load_model
from keras.preprocessing.image import *
import h5py
import json
#测试提交json文件的生成
def actop1(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,1)

def actop3(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,3)

def actop5(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,5)


X_test = []
for filename in ["Xception_test_big.h5"]:    
    with h5py.File(filename, 'r') as h:
    	X_test.append(np.array(h['test']))
X_test = np.concatenate(X_test, axis=1)

model=load_model('My_trained_xcep_big.h5')
y_pred = model.predict(X_test)

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory('/home/ross/Documents/AIC/test1', (672,672), shuffle=False, batch_size=32)
m=7040
out=[]
for i in range(m):
    y=y_pred[i]
    label=[]
    dic={}
    num=3
    while num>0:
        label.append(int(np.argmax(y)))
        y[np.argmax(y)]=0
        num-=1
    dic['image_id']=test_generator.filenames[i][5:]
    dic['label_id']=label
    out.append(dic)
with open('submit.json','w') as f:
    json.dump(out,f)
    f.flush()
    f.close()