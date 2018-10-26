import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

'''
训练集数据扩增DataAugmentation()
'''
def DataAugmentation(path):
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    classes=os.listdir(path)
    for class_name in classes:
        tmp_path = path + '/' + class_name
        imgs=os.listdir(tmp_path)
        for pic in imgs:
            img = load_img(tmp_path + '/' + pic)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1,save_to_dir=tmp_path, save_prefix=pic.split('.')[0], save_format='jpg'):
                i += 1
                if i > 1:
                    break
'''
测试集数据扩增testAugmentation()，测试集要增强2次，分开独立保存。每次要修改save_to_dir='E:/PyData/BaiDuDog/test_kz_2'分别为test_kz_1和test_kz_2
'''
def testAugmentation(path,subpath):
    datagen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    imgs=os.listdir(path)
    for pic in imgs:
        img = load_img(path + '/' + pic)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1,save_to_dir=subpath, save_prefix=pic.split('.')[0], save_format='jpg'):
            i += 1
            if i > 0:
                break
if __name__=="__main__":
    print("Augmentate train...")
    DataAugmentation("/home/ross/Documents/AIC/train")
    print("Augmentate validation...")
    DataAugmentation("/home/ross/Documents/AIC/validation")
    print("Augmentate test...")
    testAugmentation("/home/ross/Documents/AIC/test1/test",'/home/ross/Documents/AIC/test2/test')#注意名字保持一致
    testAugmentation("/home/ross/Documents/AIC/test1/test",'/home/ross/Documents/AIC/test3/test')#注意名字保持一致
    print("Augmentate over...")
