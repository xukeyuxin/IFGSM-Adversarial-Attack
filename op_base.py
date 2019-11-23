import csv
import os
import cv2
import tensorflow as tf
import numpy as np

class data(object):
    def __init__(self,args):
        self.__dict__ = args.__dict__
        # self.image_list = []
        # self.class2image = {}
        self.classNameList = []

        ### [n01440764 , n01443537 ]
        with open(self.classFile,'r' ) as f_r:
            for line in f_r:
                className = line.strip().split(',')[0]
                self.classNameList.append(className)
        # with open(self.image2class,'r') as f:
        #     for line in f:
        #         image_name,image_class = line.strip().split()
        #         image_class = int(image_class)
        #         image_name = os.path.join(self.train_image_path,image_name)
        #         self.image_list.append((image_name,image_class))
        #         self.class2image.setdefault(image_class,[])
        #         self.class2image[image_class].append(image_name)

        # self.image_list = np.asarray(self.image_list)
        # self.class_num = 1000
        # self.data_size = len(self.image_list)

    def rbg2float(self,input,need_resize = True):
        return  (cv2.resize(input,(299,299)) - 127.5) / 127.5
    def load_batch(self,batch_size = 16,shuffle = False):
        while True:
            random_index = np.random.choice(range(self.data_size),batch_size)
            image,label = zip( *self.image_list[random_index] )

            yield np.asarray(image), np.asarray(label)
    
    def random_target_images(self):
        _random_label = np.random.choice(range(1,1001),self.batch_size)
        _random_image_list = []
        for _label in _random_label:
            _random_image_list.append( np.random.choice(self.class2image[_label],self.one_target_image_num ) )
        return _random_label,_random_image_list
    
    def load_ImageNet_target_image(self,target):
        dirName = os.path.join(self.train_image_path,self.classNameList[int(target) - 1])
        image_list = os.listdir(dirName)
        index = 0
        for item in image_list:
            image_content = self.rbg2float(cv2.imread(os.path.join(dirName,item)))
            yield image_content, item
    
    def load_attack_image(self):
        with open(self.attack_image,'r') as f:
            reader = csv.reader(f)
            for im_p,label,target in reader:
                image_content = self.rbg2float(cv2.imread(os.path.join(self.attack_content,im_p)))
                yield image_content,label,target


class op_base(object):
    def __init__(self,args):
        self.__dict__ = args.__dict__
        self.data = data(args)



