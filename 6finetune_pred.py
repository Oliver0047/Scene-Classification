from keras.models import *
import keras
import numpy as np
from sklearn.utils import shuffle
#fine-tune模型的原验证集准确率测试
validation_path='/home/ross/Documents/AIC/validation'
model_path='My_trained_finetune_model_big.h5'
image_size=(672,672)

def test(val_path):
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(val_path, image_size,shuffle=False,batch_size=32,class_mode=None)
    y_valid = data=val_generator.classes
    model=load_model(model_path)
    prd=model.predict_generator(test_generator,test_generator.samples//32+1)
    return (prd,y_valid)

y_pred,y_valid = test(validation_path)

accuracy1=0
accuracy2=0

m=y_pred.shape[0]
for i in range(m):
    y=y_pred[i]
    fact=y_valid[i]
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