from keras.models import *
from keras.layers import *
from keras.applications.inception_resnet_v2 import *
from keras.preprocessing.image import *
import h5py
'''
训练集和验证集生成特征文件write_gap()
'''
train_path = '/home/ross/Documents/AIC/train'
validation_path = '/home/ross/Documents/AIC/validation'
test1_path='/home/ross/Documents/AIC/validation_test'
def write_gap(MODEL, gap_name, image_size, lambda_func=None):
	width = image_size[0]
	height = image_size[1]
	input_tensor = Input((height, width, 3))
	x = input_tensor
	if lambda_func:
		x = Lambda(lambda_func)(x)
	base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
	model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
	gen = ImageDataGenerator()
	train_generator = gen.flow_from_directory(train_path, image_size, batch_size=32,shuffle=False,class_mode=None)
	val_generator = gen.flow_from_directory(validation_path, image_size, batch_size=32,shuffle=False,class_mode=None)
	train = model.predict_generator(train_generator, train_generator.samples//32+1)
	valid = model.predict_generator(val_generator, val_generator.samples//32+1)
	with h5py.File(gap_name) as h:
		h.create_dataset("train", data=train)
		h.create_dataset("valid", data=valid)
		h.create_dataset("y_train", data=train_generator.classes)
		h.create_dataset("y_valid", data=val_generator.classes)
#write_gap(inception_v3.InceptionV3, "Inception_bottleneck.h5",(299, 299), inception_v3.preprocess_input)
#write_gap(xception.Xception, "Xception_bottleneck.h5",(299, 299), xception.preprocess_input)
#write_gap(resnet50.ResNet50, "resnet50_bottleneck_mid.h5",(448, 448))
#write_gap(vgg16.VGG16, "Vgg16_bottleneck_big.h5",(672,672))
#write_gap(inception_resnet_v2.InceptionResNetV2, "Inception_resv2_bottleneck_big.h5",(672,672),inception_resnet_v2.preprocess_input)
'''
#测试集生成特征文件test_write_gap()
'''
def test_write_gap(MODEL, gap_name, path, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(path, image_size, shuffle=False, batch_size=32,class_mode=None)
    test = model.predict_generator(test_generator, test_generator.samples//32+1)
    with h5py.File(gap_name) as h:
        h.create_dataset("test", data=test)
        #h.create_dataset("valid",data=test_generator.classes)#如果是生成原验证集的特征量，请加上这一句
#test_write_gap(inception_v3.InceptionV3, "InceptionV3_test.h5",test_path,(299, 299), inception_v3.preprocess_input)
#test_write_gap(vgg16.VGG16, "vgg16_test.h5",test1_path,(672,672))
#test_write_gap(resnet50.ResNet50, "ResNet50_val_big.h5",test1_path,(672,672))
#test_write_gap(vgg16.VGG16, "Vgg16_test_big.h5",test1_path,(672,672))
#test_write_gap(xception.Xception, "Xception_test_big.h5",test1_path,(672,672),xception.preprocess_input)
#test_write_gap(inception_resnet_v2.InceptionResNetV2, "Inception_resv2_test_big.h5",test1_path,(672,672),inception_resnet_v2.preprocess_input)
