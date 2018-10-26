# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
#将数据做切割
def img_split(la,path,size=200,num=80):
    for j in range(num):
        dir_cls = '%s/%02d'%(path, j)
        print(dir_cls)
        lst_img = os.listdir(dir_cls)
        for n_img in lst_img:
            n_img_path = "%s/%s"%(dir_cls, n_img)
            if os.path.isdir(n_img_path):
                continue
            print(n_img_path)
            
            img = plt.imread(n_img_path)        
            nrow, ncol, nlay = img.shape                  
            num_row = nrow//size
            num_col = ncol//size
    
            for i in range(num_row):
                for j in range(num_col):
                    ni_start = i*size
                    nj_start = j*size
                    img1 = img[ni_start:ni_start+size, nj_start:nj_start+size, :nlay]
                    name_img = "%s_%d%d%s"%(n_img_path.split('.')[0],i,j,'.jpg')
                    #print(name_img)
                    plt.imsave(name_img, img1)

if __name__=='__main__':
    print("Training set is spliting...")
    img_split('/home/ross/Documents/AIC/train',200,80)
    print("validation set is spliting...")
    img_split('/home/ross/Documents/AIC/validation',200,80)

