import csv
import os
import cv2
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm

class data(object):
    def __init__(self,args):
        self.__dict__ = args.__dict__
        self.classNameList = []
        self.fineune_data = []
        self.name2class = {}

        with open(self.classFile,'r' ) as f_r:
            index = 1
            for line in f_r:
                className = line.strip().split(',')[0]
                self.classNameList.append(className)
                self.name2class[className] = index
                index += 1
        
        self.class_nums = len(self.name2class)

    def rbg2float(self,input,need_resize = True):
        return  (cv2.resize(input,(299,299)) - 127.5) / 127.5
        
    def preprocess(self,img):
        rawH = float(img.shape[0])
        rawW = float(img.shape[1])
        newH = 320.0
        newW = 320.0
        test_crop = 299.0 

        if rawH <= rawW:
            newW = (rawW/rawH) * newH
        else:
            newH = (rawH/rawW) * newW
        img = cv2.resize(img, (int(newW), int(newH)))
        img = img[int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop),int((newW-test_crop)/2):int((newW-test_crop)/2)+int(test_crop)]
        # img = img[...,::-1]

        return img

    def make_label(self,image_label):
        batch_data = []
        for item in image_label:
            empty = np.zeros([self.class_nums])
            int_item = int(item) - 1
            empty[int_item] = 1
            batch_data.append(empty)
        return np.asarray(batch_data)

    def load_batch(self,batch_size = 16,shuffle = False):
        while True:
            random_index = np.random.choice(range(self.data_size),batch_size)
            image,label = zip( *self.image_list[random_index] )

            yield np.asarray(image), np.asarray(label)
    
    def load_fineune_data(self):
        image_dirs = [ dir_name for dir_name in os.listdir(self.train_image_path) if os.path.isdir(os.path.join(self.train_image_path,dir_name)) ]
        print(image_dirs)
        for _dir_name in tqdm(image_dirs):
            dis_path = os.path.join(self.train_image_path,_dir_name)
            self.fineune_data += [ os.path.join(_dir_name,path) for path in os.listdir(dis_path) ]

    def get_fineune_generator(self):
        for i in tqdm(range(len(self.fineune_data) // self.batch_size)):
            start_index = i * self.batch_size
            end_index = min( (i + 1) * self.batch_size, len(self.fineune_data) - 1)
            batch_data = self.fineune_data[start_index:end_index]
            batch_label =  [ self.name2class[path.split('/')[0]] for path in batch_data ]
            yield np.asarray( [ self.rbg2float(cv2.imread(os.path.join(self.train_image_path,path))) for path in batch_data ] ), self.make_label(batch_label)

    def shuffle(self):
        random.shuffle(self.fineune_data)

    def load_ImageNet_target_image(self,target):
        dirName = os.path.join(self.train_image_path,self.classNameList[int(target) - 1])
        image_list = os.listdir(dirName)
        index = 0
        for item in image_list:
            image_content = self.preprocess(self.rbg2float(cv2.imread(os.path.join(dirName,item))))
            yield image_content, item
    
    def load_attack_image(self):
        with open(self.attack_image,'r') as f:
            reader = csv.reader(f)
            for im_p,label,target in reader:
                image_content = self.preprocess(self.rbg2float(cv2.imread(os.path.join(self.attack_content,im_p))))
                yield im_p,image_content,label,target


class op_base(object):
    def __init__(self,args):
        self.__dict__ = args.__dict__
        self.data = data(args)



