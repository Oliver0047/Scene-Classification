#coding:utf-8
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import *
from keras.applications import *
from keras.layers import *
from keras.metrics import *
from keras.preprocessing.image import *
#fine-tune
def actop1(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,1)

def actop3(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,3)

def actop5(ytrue,ypred):
    return keras.metrics.top_k_categorical_accuracy(ytrue,ypred,5)

model_path='/home/ross/Documents/AIC/My_trained_model.h5'
train_path='/home/ross/Documents/AIC/train'
validation_path='/home/ross/Documents/AIC/validation'
image_size=(224,224)

top_model=load_model(model_path)
x1=Input((224,224, 3))
base_model=resnet50.ResNet50(input_tensor=x1, weights='imagenet', include_top=False)
x=GlobalAveragePooling2D()(base_model.output)
x = normalization.BatchNormalization()(x)
x = Dense(2048, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(80, activation='softmax')(x)
model=Model(base_model.input,x)

length=len(model.layers)
for i in range(length):
    if i<=168:
        model.layers[i].trainable=False
    else:
        model.layers[i].trainable=True
        if i>=175:
            model.layers[i].set_weights(top_model.layers[i-174].get_weights())

gen = ImageDataGenerator()
train_generator = gen.flow_from_directory(train_path, image_size,batch_size=32)
val_generator = gen.flow_from_directory(validation_path, image_size, batch_size=32)

model.compile(optimizer='Adadelta',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs = 3,
    steps_per_epoch = train_generator.samples//32+1,
    workers=10,
    shuffle=True,
    validation_data = val_generator, validation_steps = val_generator.samples//32+1)

model.save("model_finetuning.h5")

