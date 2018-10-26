import h5py
import numpy as np
import keras
from sklearn.utils import shuffle
from keras.metrics import *
from keras.preprocessing.image import *
from keras.optimizers import *
from keras.callbacks import TensorBoard
def actop1(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,1)

def actop3(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,3)

def actop5(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,5)

'''
载入特征向量load_feature()
'''
np.random.seed(2017)
#bottle=['Inception_resv2_bottleneck_big.h5','ResNet50_bottleneck_big.h5','Vgg16_bottleneck_big.h5']
bottle=['E:/AIC/vgg16_big/Vgg16_bottleneck_big.h5']
X_train = []
X_valid = []
for filename in bottle:
    with h5py.File(filename, 'r') as h:
    	X_train.append(np.array(h['train']))
    	y_train = np.array(h['y_train'])
    	X_valid.append(np.array(h['valid']))
    	y_valid = np.array(h['y_valid'])
X_train = np.concatenate(X_train, axis=1)
y_train = keras.utils.to_categorical(y_train, 80)
X_train, y_train = shuffle(X_train, y_train)
X_valid = np.concatenate(X_valid, axis=1)
y_valid = keras.utils.to_categorical(y_valid, 80)
X_valid, y_valid = shuffle(X_valid, y_valid)
'''
构建模型
'''
from keras.models import *
from keras.layers import *
np.random.seed(2017)

input_tensor = Input(X_train.shape[1:])
x = normalization.BatchNormalization()(input_tensor)
x = Dense(2048, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(80, activation='softmax')(x)
model = Model(input_tensor, x)
adam_op = Adam(lr=0.0001, beta_1=0.9,beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam_op,loss='categorical_crossentropy',metrics=['accuracy'])
'''
训练
'''
model.fit(X_train, y_train, batch_size=128, epochs=3, validation_data=(X_valid,y_valid),callbacks=[TensorBoard(log_dir='./tmp/log')])
model.save('My_trained_multiple.h5')


