#coding:utf-8
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import *
from keras.applications import *
from keras.layers import *
from keras.metrics import *
from keras.preprocessing.image import *
import pandas as pd
import numpy as np
import json

test1_path='/home/ross/Documents/AIC/test1'
test2_path='/home/ross/Documents/AIC/test2'
test3_path='/home/ross/Documents/AIC/test3'
image_size=(224,224)

def test(test_path):
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(test_path, image_size,shuffle=False,batch_size=32,class_mode=None)
    model=load_model('model_finetuning.h5')
    prd=model.predict_generator(test_generator,220)
    return prd

y1=test(test1_path)
y2=test(test2_path)
y3=test(test3_path)
pred=y1+y2+y3
y_pred=pred/3

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory(test1_path, (224,224), shuffle=False, batch_size=32, class_mode=None)
m=y_pred.shape[0]
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
