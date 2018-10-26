# -*- coding: utf-8 -*-
#autor:Oliver0047
#devide images into different kinds
import json
import os

train_path='/home/ross/Documents/AIC/train'
val_path='/home/ross/Documents/AIC/validation'

T_label_path='/home/ross/Documents/AIC/scene_train_annotations_20170904.json'
V_label_path='/home/ross/Documents/AIC/original_val/scene_validation_annotations_20170908.json'

class EmptyException(Exception):
    pass

#将数据按照标签分类
def data_classify(path,labels):
    if os.path.isdir(path) and os.path.getsize(path)>0:
        for j in labels:
            img=path+'/'+j['image_id']
            dir_path=path+'/'+"%02d"%(int(j['label_id']))
            target_path=dir_path+'/'+j['image_id']
            if os.path.exists(img):
                if os.path.exists(dir_path):
                    os.rename(img,target_path)
                else:
                    os.mkdir(dir_path)
                    os.rename(img,target_path)
            else:
                print(j,'\n图片不存在,需要下载!')
    else:
        raise EmptyException
        
def data_handler():
    print('###开始进行训练数据和验证数据分类###')
    train_labels=json.load(open(T_label_path,'r',encoding='utf-8'))
    data_classify(train_path,train_labels)
    validation_labels=json.load(open(V_label_path,'r',encoding='utf-8'))
    data_classify(val_path,validation_labels)
    print('###训练数据和验证数据分类结束###')
