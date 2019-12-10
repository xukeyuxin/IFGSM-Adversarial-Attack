import tensorflow as tf
from model.inception.inception import inception
from model.vgg.vgg_16 import VGG16
from model.resnet.resnet import resnet
from model.resnet_telent import ResNet
import numpy as np
from functools import reduce
import json
import pickle 
from op_base import op_base
import os
import cv2
import math
from tqdm import tqdm
from tensorflow.contrib.layers import xavier_initializer
from guass import GaussianBlur
import numpy as np
import random

class Classify(op_base):
    def __init__(self,sess,args):
        op_base.__init__(self,args)
        self.sess = sess
        self.summary = []
        self.random_size_step = 0
        # self.model_list = ['inception_v4','inception_v3','inception_res','resnet_50','resnet_101','resnet_152','resnet_tel']
        # self.model_list = ['inception_v4','inception_res','resnet_tel']
        self.model_list = ['inception_res']
        self.input_images = tf.placeholder(tf.float32,shape = [None,self.image_height,self.image_weight,3])
        self.input_blur_images = tf.placeholder(tf.float32,shape = [None,self.image_height,self.image_weight,3])
        self.target_feature = tf.placeholder(tf.float32,shape = [2048])
        self.target_label = tf.placeholder(tf.int32,shape = [None,1000])
        self.label_feature = tf.placeholder(tf.float32,shape = [2048])
        self.label_label = tf.placeholder(tf.int32,shape = [None,1000])
        self.mask = tf.placeholder(tf.float32,shape = [1,self.image_height,self.image_weight,1])
        self.index = tf.placeholder(tf.int32,shape = [])
        # self.stop_value = tf.placeholder(tf.float32, shape = [])
        self.gaussian_blur = GaussianBlur()

        attack = True if self.action == 'attack' else False
        is_training = True if self.action == 'train' else False
        if(attack):
            self.init_noise()
            # self.load_both_model()
        else:

            if(self.model_type == 'inception_v4'):
                self.model = inception(self.input_images,is_training = is_training)
                self.model.inception_v4()
                self.save_model = 'model/inception/model/inception_v4'
                if(is_training):
                    self.pre_model = 'model/inception/pretrain/inception_v4.ckpt'     
                else:
                    self.pre_model = 'model/inception/model/inception_v4/inception_v4.ckpt'

                self.variables_to_restore, self.variables_to_train = self.model.get_train_restore_vars()
                self.init_model()
                
                self.saver = tf.train.Saver(self.variables_to_restore)
                self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)
            
            elif(self.model_type == 'inception_v3'):
                self.model = inception(self.input_images,is_training = is_training)
                self.model.inception_v3()
                self.save_model = 'model/inception/model/inception_v3'
                if(is_training):
                    self.pre_model = 'model/inception/pretrain/inception_v3.ckpt'     
                else:
                    self.pre_model = 'model/inception/model/inception_v3/inception_v3.ckpt'

                
                self.init_model()
                self.variables_to_restore, self.variables_to_train = self.model.get_train_restore_vars()
                self.saver = tf.train.Saver(self.variables_to_restore)
                self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)

            elif(self.model_type == 'inception_res'):
                self.model = inception(self.input_images,is_training = is_training)
                self.model.inception_res()
                self.save_model = 'model/inception/model/inception_res'
                if(is_training):
                    self.pre_model = 'model/inception/pretrain/inception_resnet_v2.ckpt'   
                else:
                    self.pre_model = 'model/inception/model/inception_res/inception_res.ckpt'

                self.init_model()
                self.variables_to_restore, self.variables_to_train = self.model.get_train_restore_vars()           
                self.saver = tf.train.Saver(self.variables_to_restore)
                self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)
            
            elif(self.model_type == 'vgg_16'):
                self.model = VGG16(self.input_images,is_training = is_training)
                self.model()
                self.save_model = 'model/vgg/model'
                self.pre_model = 'model/vgg/pretrain/vgg_16.ckpt'
                self.init_model()
                self.variables_to_restore, self.variables_to_train = self.model.get_train_restore_vars()
                self.saver = tf.train.Saver(self.variables_to_restore)
                self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)

            elif(self.model_type == 'resnet_tel'):
                self.model = ResNet(self.input_images,is_training = is_training)
                self.model()
                self.pre_model = 'model/resnet_tel/resnet_tel.ckpt'
                self.init_model()
                
                variables_to_restore = self.model.get_restore_variable()
                print(len(variables_to_restore))
                # print()
                self.saver = tf.train.Saver(variables_to_restore)

                # self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)

                # self.resnet_tel_model = ResNet(input,is_training = False)
                # self.resnet_tel_model()
                # self.resnet_tel_pre_model = 'model/resnet_tel/resnet.ckpt'
                
                # self.saver = tf.train.Saver(variables_to_restore)
                # self.resnet_tel_saver = tf.train.Saver(variables_to_restore)


            elif(self.model_type == 'resnet_50'):
                self.model = resnet(self.input_images,is_training = is_training)
                self.model.resnet_50()
                self.save_model = 'model/resnet/model/resnet_50'
                if(is_training):
                    self.pre_model = 'model/resnet/pretrain/resnet_v2_50.ckpt'   
                else:
                    self.pre_model = 'model/resnet/model/resnet_50/resnet_50.ckpt'
                self.init_model()
                self.variables_to_restore, self.variables_to_train = self.model.get_train_restore_vars()
                print('find variable shape %s' % (len(self.variables_to_restore + self.variables_to_train)))
                self.saver = tf.train.Saver(self.variables_to_restore,max_to_keep = 1)   
                self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)  

            elif(self.model_type == 'resnet_101'):
                self.model = resnet(self.input_images,is_training = is_training)
                self.model.resnet_101()
                self.save_model = 'model/resnet/model/resnet_101'
                if(is_training):
                    self.pre_model = 'model/resnet/pretrain/resnet_v2_101.ckpt'   
                else:
                    self.pre_model = 'model/resnet/model/resnet_101/resnet_101.ckpt'
                self.init_model()
                self.variables_to_restore, self.variables_to_train = self.model.get_train_restore_vars()
                self.saver = tf.train.Saver(self.variables_to_restore,max_to_keep = 1)   
                self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)  

            elif(self.model_type == 'resnet_152'):
                self.model = resnet(self.input_images,is_training = is_training)
                self.model.resnet_152()
                self.save_model = 'model/resnet/model/resnet_152'
                if(is_training):
                    self.pre_model = 'model/resnet/pretrain/resnet_v2_152.ckpt'   
                else:
                    self.pre_model = 'model/resnet/model/resnet_152/resnet_152.ckpt'
                self.init_model()
                self.variables_to_restore, self.variables_to_train = self.model.get_train_restore_vars()
                self.saver = tf.train.Saver(self.variables_to_restore,max_to_keep = 1)   
                self.saver_store = tf.train.Saver(self.variables_to_restore + self.variables_to_train,max_to_keep = 1)  
 

            print('finish load %s' % self.model_type)
        
        self.attack_generator = self.data.load_attack_image()
        self.target_generator = self.data.load_ImageNet_target_image
    def get_restore_var(self,exclusion):
        if( isinstance(exclusion,str)):
            return [ var for var in tf.trainable_variables() if var.op.name.startswith(exclusion) ]
        elif(isinstance(exclusion,list)):
            var_list = []
            for var in tf.trainable_variables():
                for item_exclusion in exclusion:
                    if(var.op.name.startswith(item_exclusion)):
                        var_list.append(var)
            return var_list

    def build_all_graph(self,input):
        
        for item in self.model_list:
            if(item == 'inception_v4'):
                self.inception_v4_model = inception(input,is_training = False)
                self.inception_v4_model.inception_v4()
                self.inception_v4_pre_model = 'model/inception/model/inception_v4/inception_v4.ckpt'
                variables_to_restore = self.inception_v4_model.get_train_restore_vars('InceptionV4')[0]
                self.inception_v4_saver = tf.train.Saver(variables_to_restore)
            elif(item == 'inception_v3'):
                self.inception_v3_model = inception(input,is_training = False)
                self.inception_v3_model.inception_v3()
                self.inception_v3_pre_model = 'model/inception/model/inception_v3/inception_v3.ckpt'
                variables_to_restore = self.inception_v3_model.get_train_restore_vars('InceptionV3')[0]
                self.inception_v3_saver = tf.train.Saver(variables_to_restore)
            elif(item == 'inception_res'):
                self.inception_res_model = inception(input,is_training = False)
                self.inception_res_model.inception_res()
                self.inception_res_pre_model = 'model/inception/model/inception_res/inception_res.ckpt'
                variables_to_restore = self.inception_res_model.get_train_restore_vars('InceptionResnetV2')[0]
                self.inception_res_saver = tf.train.Saver(variables_to_restore)

            elif(item == 'resnet_tel'):
                self.resnet_tel_model = ResNet(input,is_training = False)
                self.resnet_tel_model()
                self.resnet_tel_pre_model = 'model/resnet_tel/resnet_tel.ckpt'
                variables_to_restore = self.resnet_tel_model.get_restore_variable()
                self.saver = tf.train.Saver(variables_to_restore)
                self.resnet_tel_saver = tf.train.Saver(variables_to_restore)

            elif(item == 'resnet_50'):
                self.resnet_50_model = resnet(input,is_training = False)
                self.resnet_50_model.resnet_50()
                self.resnet_50_pre_model = 'model/resnet/model/resnet_50/resnet_50.ckpt'
                variable_mix = []
                for scope in ['resnet_v2_50','resnet_50']:
                    variables_to_restore, variables_to_train = self.resnet_50_model.get_train_restore_vars(scope)
                    variable_mix += variables_to_restore
                self.resnet_50_saver = tf.train.Saver(variable_mix)            
            elif(item == 'resnet_101'):
                self.resnet_101_model = resnet(input,is_training = False)
                self.resnet_101_model.resnet_101()
                self.resnet_101_pre_model = 'model/resnet/model/resnet_101/resnet_101.ckpt'
                variable_mix = []
                for scope in ['resnet_v2_101','resnet_101']:
                    variables_to_restore, variables_to_train = self.resnet_101_model.get_train_restore_vars(scope)
                    variable_mix += variables_to_restore
                self.resnet_101_saver = tf.train.Saver(variable_mix)    
            elif(item == 'resnet_152'):
                self.resnet_152_model = resnet(input,is_training = False)
                self.resnet_152_model.resnet_152()
                self.resnet_152_pre_model = 'model/resnet/model/resnet_152/resnet_152.ckpt'
                variable_mix = []
                for scope in ['resnet_v2_152','resnet_152']:
                    variables_to_restore, variables_to_train = self.resnet_152_model.get_train_restore_vars(scope)
                    variable_mix += variables_to_restore
                self.resnet_152_saver = tf.train.Saver(variable_mix)    

    def init_all_var(self):
        self.sess.run(tf.global_variables_initializer())
        for item in self.model_list:
            if(item == 'inception_v4'):
                self.inception_v4_saver.restore(self.sess,self.inception_v4_pre_model)
            elif(item == 'inception_v3'):
                self.inception_v3_saver.restore(self.sess,self.inception_v3_pre_model)
            elif(item == 'inception_res'):
                self.inception_res_saver.restore(self.sess,self.inception_res_pre_model)
            elif(item == 'resnet_50'):
                self.resnet_50_saver.restore(self.sess,self.resnet_50_pre_model)
            elif(item == 'resnet_tel'):
                self.resnet_tel_saver.restore(self.sess,self.resnet_tel_pre_model)
            elif(item == 'resnet_101'):
                self.resnet_101_saver.restore(self.sess,self.resnet_101_pre_model)
            elif(item == 'resnet_152'):
                self.resnet_152_saver.restore(self.sess,self.resnet_152_pre_model)
        print('restore finish')

    def convert(self,input):
        return tf.convert_to_tensor(input)
    
    def get_vars(self,name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = name)

    def average_gradients(self,tower_grads):
        """ Calculate the average gradient of shared variables across all towers. """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for grad, var in grad_and_vars:
                grads.append(tf.expand_dims(grad, 0))
            # Average over the 'tower' dimension.
            gradient = tf.reduce_mean(tf.concat(axis=0, values=grads), 0)
            v = grad_and_vars[0][1]
            grad_and_var = (gradient, v)
            average_grads.append(grad_and_var)
        return average_grads

    def init_model(self,noise = False):
        self.global_step = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),dtype= tf.int64,
                            trainable=False)
        lr = tf.train.exponential_decay(
            self.lr,
            self.global_step,
            self.lr_decay_step,
            self.lr_decay_factor,
            staircase=True)
        self.summary.append(tf.summary.scalar('lr',lr))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, momentum=self.opt_momentum)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        

    def graph(self,logit):
        loss_softmax = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logit,labels = self.input_label))
        tf.identity(loss_softmax,name = 'loss_softmax')
        self.summary.append(tf.summary.scalar('loss_softmax',loss_softmax))
        l2_regularization = self.weight_decay * tf.add_n( [ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bn' not in v.name ] )
        l2_regularization_bn = 0.1 * self.weight_decay * tf.add_n(  [ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bn' in v.name ]  )
        loss = loss_softmax + l2_regularization + l2_regularization_bn
        grads = self.optimizer.compute_gradients(loss,tf.trainable_variables())
        return grads

    def normal_2(self,input):
        return  input / ( np.sqrt( np.sum(np.square(input) ) ) )
    
    def tensor_normal_2(self,input):
        return input / tf.sqrt(tf.reduce_sum(tf.square(input)))

    def xavier_initializer(self,shape, gain = 1.):
        if(len(shape) == 4):
            fan_in = reduce( np.multiply, shape[1:] )  # 输入通道
            fan_out = reduce( np.multiply, shape[1:] )  # 输出通道
        variation = (2/( fan_in +fan_out)) * gain
        std = math.sqrt(variation)
        result = np.random.normal(0,std,shape)
        return result
       
    def find_sim(self):
        def cal(logits, features, target_index):
            logits = np.asarray(logits)
            logits_value = logits[:,target_index - 1]
            arg = np.argmax(logits_value)
            features = np.asarray(features)
            return features[arg]

        root_dir = os.path.join('data/feature')
        for item in map(str,range(1,1218)):
            im_p,image_content,label,target = next(self.attack_generator)
            logits_1 = []
            features_1 = []

            target = int(target)

            single = open(os.path.join(root_dir,item,'single_label.pickle'),'rb')
            single.close()

            with open(os.path.join(root_dir,item,'target_feature_logit.pickle'),'rb') as f1,open(os.path.join(root_dir,item,'label_feature_logit.pickle'),'rb') as f2, open(os.path.join(root_dir,item,'target_best_feature.pickle'),'ab+') as f_w:
                while True:
                    try:
                        _path,feature,logit = pickle.load(f1)
                        logits_1.append(logit)
                        features_1.append(feature)
                    except EOFError:
                        print('load finish')
                        break

                target_feature = cal(logits_1, features_1,target)
                pickle.dump(target_feature,f_w)

                return


    def tv_loss(self,input_t):

        temp1 = tf.concat( [ input_t[:,1:,:,:], tf.expand_dims(input_t[:,-1,:,:],axis = 1)],axis = 1 )
        temp2 = tf.concat( [ input_t[:,:,1:,:], tf.expand_dims(input_t[:,:,-1,:],axis = 2)],axis = 2 )
        temp = (input_t - temp1)**2 +  (input_t - temp2)**2
        return tf.reduce_sum(temp)
    
    def pre_noise(self,mask):
        return mask * self.gaussian_blur(self.tmp_noise) 
    
    def write_noise(self,mask,noise):
        return mask * self.gaussian_blur(noise) 
    
    def float2rgb(self,input):
        return input * 127.5 + 127.5
    
    def resize(self,input):
        return np.reshape(input,(299,299,3))

    def update_op(self,new_noise):
        return tf.assign(self.noise,new_noise)

    def make_feed_dict(self,input_image,target_label,label_label,mask,index):
        # input_blur_images =  np.expand_dims(cv2.resize(cv2.resize(np.squeeze(input_image),(224,224)),(299,299)),0)
        # return {self.input_images:input_image,self.input_blur_images:input_blur_images,self.target_label:target_label,self.label_label:label_label,self.mask:mask,self.index:index} 
        # return {self.input_images:input_image,self.target_feature:target_feature,self.label_feature:label_feature,self.mask:mask,self.index:index} 
        return {self.input_images:input_image,self.target_label:target_label,self.label_label:label_label,self.mask:mask,self.index:index}
    def init_noise(self):
        ## init
        tmp_noise_init = self.xavier_initializer([1,299,299,3])
        grad_init = 0.
        self.tmp_noise = tf.get_variable('noise',shape = [1,299,299,3], initializer= tf.constant_initializer(tmp_noise_init))
        # self.tmp_noise = tf.get_variable('noise',shape = [1,299,299,3], initializer= tf.constant_initializer(0.))
        self.v1_grad = tf.get_variable('noise_grad',shape = [1,299,299,3],initializer= tf.constant_initializer(grad_init))
    def tf_assign_init(self):
        tmp_noise_init = self.xavier_initializer([1,299,299,3])
        grad_init = tf.constant(0.,shape = [1,299,299,3])
        update_tmp_noise = tf.assign(self.tmp_noise,tmp_noise_init)
        update_v1_grad = tf.assign(self.v1_grad,grad_init)
        return tf.group(update_tmp_noise,update_v1_grad)

    def tf_preprocess(self,img,lar_size,out_size):
        rawH = self.image_height
        rawW = self.image_weight

        newH = lar_size
        newW = lar_size
        test_crop = out_size

        img = tf.image.resize_images(img,[newH,newW])  
        img = img[:,int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop),int((newW-test_crop)/2):int((newW-test_crop)/2)+int(test_crop)]

        return img

    def attack_graph(self,lr = 0.1,momentum = 0.5):
        def entropy_loss(logits):
            label_loss_cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logits))
            target_loss_cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logits))
            self.target_loss = target_loss_cross_entropy
            self.label_loss = label_loss_cross_entropy
            alpha1 = tf.cond(label_loss_cross_entropy > 50.,lambda: 0.,lambda: 1.)
            alpha2 = tf.cond(target_loss_cross_entropy < 1,lambda: 0.,lambda: 1.)

            return  alpha2 * target_loss_cross_entropy - alpha1 * label_loss_cross_entropy

        ### random loss
        # random_list = [self.combine_images_320_299,self.combine_images,self.combine_images_blur_320_299,self.combine_images_blur]
        # def random_loss(combine_image):
        #     logit = self.model(combine_image)
        #     loss = entropy_loss(logit)
        #     return loss
        # random_choise = tf.random_uniform([],0,4,dtype = tf.int32)
        # image_choose = tf.gather(random_list,random_choise)
        # loss_total = random_loss(image_choose)
        # self.combine_images_320_299 = tf.clip_by_value(self.tf_preprocess(self.input_images,320,299) + tmp_noise,-1.,1.)
        # self.combine_images_blur_320_299 = tf.clip_by_value(self.tf_preprocess(self.input_blur_images,320,299) + tmp_noise,-1.,1.)
        # self.combine_images_blur = tf.clip_by_value(self.input_blur_images + tmp_noise,-1.,1.)

        def get_label_index(label):
            _label = tf.squeeze(label)
            _label_max = tf.argsort(_label)[-1:]
            # assert _label_max.shape == []
            return _label_max

        def large_alpha_r(input,alpha = 5.):
            _softmax_input = tf.squeeze(tf.nn.softmax(input))
            _target_index = get_label_index(self.target_label)
            _target_logit = tf.gather(_softmax_input,_target_index)
            _second_value = tf.sort(_softmax_input)[-2]
            r = tf.cond( tf.squeeze(_target_logit) >= alpha * tf.squeeze(_second_value),lambda: 0.,lambda: 1.)
            return r

            
        # def cut_320_299(input,newH = 320,test_crop = 299):
        #     newW = newH
        #     # input = tf.squeeze(input)
        #     new_image = tf.image.resize_images(input,(newH,newH))
        #     new_image = new_image[:,int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop),int((newW-test_crop)/2):int((newW-test_crop)/2)+int(test_crop)]
        #     new_image = tf.reshape(new_image,[-1,test_crop,test_crop,3])
        #     print(new_image)
        #     return new_image

        def cell_graph(logit,need_label_cross = True,need_target_cross = True):
            r_restel_tar_base = 0.
            r_restel_lab_base = 0.
            loss_resnet_tel_base = 0.
            if(need_target_cross):
                target_cross_entropy_resnet_tel= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logit)) 
                r_restel_tar_base = 1.
                self.target_cross_entropy_resnet_tel = target_cross_entropy_resnet_tel
                # r_restel_tar_base = large_alpha_r(logit) 
                loss_resnet_tel_base += r_restel_tar_base * target_cross_entropy_resnet_tel
            if(need_label_cross):
                label_cross_entropy_resnet_tel = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logit)) 
                self.label_cross_entropy_resnet_tel = label_cross_entropy_resnet_tel
                r_restel_lab_base = tf.cond( label_cross_entropy_resnet_tel < -50.,lambda: 0.,lambda: 1.)
                loss_resnet_tel_base -= r_restel_lab_base * label_cross_entropy_resnet_tel

            return loss_resnet_tel_base, r_restel_tar_base, r_restel_lab_base
            # else:
            #     loss_resnet_tel_base = r_restel_tar_base * target_cross_entropy_resnet_tel
            #     return loss_resnet_tel_base, r_restel_tar_base

        def tf_resize(input):
            # self.random_size_step += 1

            # self.new_size = tf.cast( 200 + 50 * (tf.floor(self.index / 50)), tf.int32).eval()

            # self.new_size = tuple(np.random.randint(200,400,(2)))
            # new_size = (300,300)
            new_size = tuple(tf.random_uniform((2),200,400,dtype=tf.int32).eval())
            return tf.image.resize_images(input,new_size)
            
        def item_graph(_model,need_change_channel_noise = False,newH = 299, test_crop = 299):

            if(newH != 299):
                new_image = tf.image.resize_images(self.input_images,(newH,newH))
                new_image = new_image[:,int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop),int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop)]
                new_image = tf.reshape(new_image,[1,test_crop,test_crop,3])
            else:
                new_image = self.input_images
            # new_image = tf.clip_by_value(new_image + tmp_noise,-1.,1.)
            clip_image = tf.clip_by_value(new_image + tmp_noise,-1.,1.)
            random_resize = tf_resize(clip_image)
            self.new_size = random_resize
            # print(random_resize.shape)
            logits_resnet_tel_base = _model(random_resize)
            rgb_loss, rgb_stop_t, rgb_stop_l = cell_graph(logits_resnet_tel_base)
            if(need_change_channel_noise):
                # logits_resnet_tel_bgr = _model(tf.clip_by_value(new_image + tmp_noise,-1.,1.)[...,::-1])
                # bgr_loss, bgr_stop_t, bgr_stop_l = cell_graph(logits_resnet_tel_bgr)

                logits_resnet_tmp = _model(tf.clip_by_value(tmp_noise,-1.,1.))
                noise_loss, noise_stop_t,noise_stop_l = cell_graph(logits_resnet_tmp,need_label_cross = False,need_target_cross = True)

                return rgb_loss + 1. * noise_loss, rgb_stop_t + noise_stop_t, rgb_stop_l + noise_stop_l
                # return rgb_loss + bgr_loss + 1. * noise_loss, rgb_stop_t + bgr_stop_t + noise_stop_t, rgb_stop_l + bgr_stop_l + noise_stop_l

            else:
                return rgb_loss , rgb_stop_t , rgb_stop_l

                



        def mask_gradient(grads,drop_probs = int(1 * 299 * 299),flatten_shape = [299*299,3]):
            grads = tf.squeeze(grads)
            grads_flatten = tf.reshape(grads,flatten_shape)

            gradient_mix = tf.reduce_sum(tf.abs(grads_flatten),axis = -1)

            top_op = tf.nn.top_k(gradient_mix, drop_probs)
            gradient_mask_index = tf.expand_dims(top_op.indices,-1)
            gradient_mask_zero_value = tf.tile( tf.expand_dims(tf.ones_like(top_op.values),axis = -1) , [1,3] ) 
            gradient_scatter = tf.scatter_nd(gradient_mask_index,gradient_mask_zero_value, [299*299,3])
            grads_flatten = grads_flatten * gradient_scatter
            gradient_mask = tf.reshape(grads_flatten, [-1,299,299,3])

            return gradient_mask

        ### softmax loss
        tmp_noise = self.pre_noise(self.mask)
        self.combine_images = tf.clip_by_value(self.input_images + tmp_noise,-1.,1.)
        # self.combine_images_change_channels = tf.clip_by_value(self.input_images[...,::-1] + tmp_noise,-1.,1.)
        
        # with tf.control_dependencies([self.combine_images]):
        self.build_all_graph(self.input_images)
        self.init_all_var()

        _loss_total = 0
        _stop_mix = 0.
        model_weight_length = len(self.model_list)
        for item in self.model_list:
            if(item == 'inception_v4'):
                ## inception4
                alpha1 = 1 / model_weight_length
                _loss,stop_t,stop_l = item_graph(self.inception_v4_model.inception_v4)
                _loss_total += _loss * alpha1
                _stop_mix += (stop_t + stop_l)
                self.inception_v4_stop_mix = stop_t + stop_l

            elif(item == 'inception_v3'):
                ## inception3
                alpha2 = 1 / model_weight_length
                _loss,stop_t,stop_l = item_graph(self.inception_v3_model.inception_v3)
                _loss_total += _loss * alpha2
                _stop_mix += (stop_t + stop_l)
                self.inception_v3_stop_mix = stop_t + stop_l

                # alpha2 = 1 / model_weight_length
                # logits_inception_v3 = self.inception_v3_model.inception_v3(self.combine_images)
                # # logits_inception_v3 = self.inception_v3_model.logits
                # target_cross_entropy_inception_v3 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logits_inception_v3)) 
                # label_cross_entropy_inception_v3 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logits_inception_v3)) 

                # r_inception_v3_tar = large_alpha_r(logits_inception_v3)
                # r_inception_v3_lab = tf.cond( label_cross_entropy_inception_v3 > 50.,lambda: 0.,lambda: 1.)

                # loss_inception_v3 = r_inception_v3_tar * target_cross_entropy_inception_v3 - r_inception_v3_lab * label_cross_entropy_inception_v3
                # _loss_total += loss_inception_v3 * alpha2

                # self.target_cross_entropy_inception_v3 = target_cross_entropy_inception_v3
                # self.label_cross_entropy_inception_v3 = label_cross_entropy_inception_v3

            elif(item == 'inception_res'):
                ## inception_res
                alpha3 = 1 / model_weight_length
                _loss,stop_t,stop_l = item_graph(self.inception_res_model.inception_res,need_change_channel_noise=True)

                _loss_total += _loss * alpha3
                _stop_mix += (stop_t + stop_l)

                self.inception_res_stop_mix = stop_t + stop_l
                

            elif(item == 'resnet_50'):
                ## resnet_50
                alpha4 = 1 / model_weight_length
                _loss,stop_t,stop_l = item_graph(self.resnet_50_model.resnet_50)
                _loss_total += _loss * alpha4
                _stop_mix += (stop_t + stop_l)
                # self.inception_v3_stop_mix = stop_t + stop_l


                # alpha4 = 1 / model_weight_length
                # logits_resnet_50 = self.resnet_50_model.resnet_50(self.combine_images)
                # # logits_resnet_50 = self.resnet_50_model.logits
                # target_cross_entropy_resnet_50= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logits_resnet_50)) 
                # label_cross_entropy_resnet_50 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logits_resnet_50)) 

                # r_res50_tar = large_alpha_r(logits_resnet_50)
                # r_res50_lab = tf.cond( label_cross_entropy_resnet_50 > 50.,lambda: 0.,lambda: 1.)
                # loss_resnet_50 = r_res50_tar * target_cross_entropy_resnet_50 - r_res50_lab * label_cross_entropy_resnet_50
                # _loss_total += loss_resnet_50 * alpha4

                # self.target_cross_entropy_resnet_50 = target_cross_entropy_resnet_50
                # self.label_cross_entropy_resnet_50 = label_cross_entropy_resnet_50

            elif(item == 'resnet_101'):
                ## resnet_101
                alpha5 = 1 / model_weight_length
                _loss,stop_t,stop_l = item_graph(self.resnet_101_model.resnet_101)
                _loss_total += _loss * alpha5
                _stop_mix += (stop_t + stop_l)

                # alpha5 = 1 / model_weight_length
                # logits_resnet_101 = self.resnet_101_model.resnet_101(self.combine_images)
                # # logits_resnet_101 = self.resnet_101_model.logits
                # target_cross_entropy_resnet_101= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logits_resnet_101)) 
                # label_cross_entropy_resnet_101 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logits_resnet_101)) 

                # r_res101_tar = large_alpha_r(logits_resnet_101)
                # r_res101_lab = tf.cond( label_cross_entropy_resnet_101 > 50.,lambda: 0.,lambda: 1.)
                # loss_resnet_101 = r_res101_tar * target_cross_entropy_resnet_101 - r_res101_lab * label_cross_entropy_resnet_101
                # _loss_total += loss_resnet_101 * alpha5


                # self.target_cross_entropy_resnet_101 = target_cross_entropy_resnet_101
                # self.label_cross_entropy_resnet_101 = label_cross_entropy_resnet_101
                

            elif(item == 'resnet_152'):
                ## resnet_152
                alpha6 = 1 / model_weight_length
                _loss,stop_t,stop_l = item_graph(self.resnet_152_model.resnet_152)
                _loss_total += _loss * alpha6
                _stop_mix += (stop_t + stop_l)


                # alpha6 = 1 / model_weight_length
                # logits_resnet_152 = self.resnet_152_model.resnet_152(self.combine_images)
                # # logits_resnet_152 = self.resnet_152_model.logits
                # target_cross_entropy_resnet_152= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logits_resnet_152)) 
                # label_cross_entropy_resnet_152 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logits_resnet_152)) 
                
                # r_res152_tar = large_alpha_r(logits_resnet_152)
                # r_res152_lab = tf.cond( label_cross_entropy_resnet_152 > 50.,lambda: 0.,lambda: 1.)
                # loss_resnet_152 = r_res152_tar * target_cross_entropy_resnet_152 - r_res152_lab * label_cross_entropy_resnet_152
                # _loss_total += loss_resnet_152 * alpha6

                # self.target_cross_entropy_resnet_152 = target_cross_entropy_resnet_152
                # self.label_cross_entropy_resnet_152 = label_cross_entropy_resnet_152

            elif(item == 'resnet_tel'):
                alpha7 = 1 / model_weight_length
                _loss,stop_t,stop_l = item_graph(self.resnet_tel_model,need_change_channel_noise = True)
                _loss_total += _loss * alpha7
                _stop_mix += (stop_t + stop_l)
                self.inception_tel_stop_mix = stop_t + stop_l

        # self.inception_stop_value = r_inception_v4_tar + r_inception_v4_lab + r_inception_res_tar + r_inception_res_lab
        # self.resnet_stop_value = _stop_mix
        # self.mix_stop = self.inception_stop_value + self.resnet_stop_value
        self.mix_stop = _stop_mix

        # # ### logits

        # logits_base = self.model(self.combine_images)
        # logits_base_320_299 = self.model(self.combine_images_320_299)
        # logits_blur = self.model(self.combine_images_blur)
        # logits_blur_320_299 = self.model(self.combine_images_blur_320_299)

        # base_loss = entropy_loss(logits_base)
        # base_320_299_loss = entropy_loss(logits_base_320_299)
        # blur_loss = entropy_loss(logits_blur)
        # blur_320_299_loss = entropy_loss(logits_blur_320_299)
        # loss_total = base_loss + base_320_299_loss + blur_loss + blur_320_299_loss

        # ### feature
        # with_noise_feat = self.tensor_normal_2(self.model(combine_images))
        # loss_feat_1 = tf.reduce_sum(self.label_feature * with_noise_feat) 
        # loss_feat_2 = tf.reduce_sum(self.target_feature * with_noise_feat)

        # alpha1 = tf.cast(tf.cond(loss_feat_1 < 0.3,lambda: 0.1,lambda: 5.), tf.float32)
        # alpha2 = tf.cast(tf.cond(loss_feat_2 > 0.7,lambda: 0.1,lambda: 5.), tf.float32)
        # loss_feat = alpha1 * loss_feat_1 - alpha2 * loss_feat_2

        feat_grad = tf.gradients(ys = _loss_total,xs = self.tmp_noise)[0] ## (299,299,3)

        loss1_grad = feat_grad * (1 - momentum) + self.v1_grad * momentum
        
        ### weight loss
        loss_l2 = tf.sqrt(tf.reduce_sum(tmp_noise**2))
        # loss_tv = self.tv_loss(tmp_noise)

        r3 = 1.
        r3 = tf.cond(self.index > 100,lambda: r3 * 0.1,lambda: r3)
        r3 = tf.cond(self.index > 200,lambda: r3 * 0.1,lambda: r3)

        # loss_weight = r3 * 0.025 * loss_l2 + r3 * 0.004 * loss_tv   
        loss_weight = r3 * 0.025 * loss_l2 

        finetune_grad = tf.gradients(loss_weight,self.tmp_noise)[0]  
        
        ### mix_grad mask
        mix_grad_mask = mask_gradient(finetune_grad + loss1_grad)

        ### finetune grad mask + l2_loss
        # loss1_grad_mask = mask_gradient(loss1_grad)
        # mix_grad_mask = loss1_grad_mask + finetune_grad

        ### gradient_mask

        update_noise = self.tmp_noise - lr * tf.sign(mix_grad_mask)
        update_noise = update_noise + tf.clip_by_value(self.input_images, -1., 1.) - self.input_images
        update_noise = tf.clip_by_value(update_noise,-0.25, 0.25)
        # update_noise = tf.clip_by_value(update_noise,-0.25, 0.25)

        self.total_loss = _loss_total
        self.loss_weight = loss_l2

        # _noise,_feat_1,_feat_2,_weight = self.sess.run([update_noise,self.loss_feat_1,self.loss_feat_2,self.loss_weight],feed_dict = {self.input_images:_image_content})
        update_value = tf.assign(self.tmp_noise,update_noise)
        update_grad = tf.assign(self.v1_grad,loss1_grad)
        return tf.group(update_value,update_grad)

    def writer(self,_image_path,write_image):
        write_image = self.float2rgb(np.squeeze(write_image))
        image_combine_with_noise = os.path.join('data','test_result','test_resize_v_res',_image_path)
        cv2.imwrite(image_combine_with_noise,write_image)

    def attack(self):
        train_op = self.attack_graph()
        hard_writer = open('hard.txt','a+')
        for _ in tqdm(range(100)):
            _image_path,_image_content,_label,_target = next(self.attack_generator)
            print('start one attack image: %s' %  _image_path)
            label_np = np.array([int(_label)])
            target_np = np.array([int(_target)])

            label_input = self.data.make_label(label_np)
            target_input = self.data.make_label(target_np)

            _image_content = np.expand_dims(_image_content ,0) # (1,299,299,3)
            mask = np.ones([1,299,299,1])
            print('start attack %s' % _image_path)
            for i in range(0,501):
                feed_dict = self.make_feed_dict(_image_content,target_input,label_input,mask,i)
                _,write_image,_weight,_loss,_t_loss,_l_loss,_new_size = self.sess.run([train_op,self.combine_images,self.loss_weight,self.total_loss,self.target_cross_entropy_resnet_tel,self.label_cross_entropy_resnet_tel,self.new_size],feed_dict = feed_dict)
                print(_loss)
                print(_t_loss)
                print(_l_loss)
                print(_new_size.shape)
                print('-----------finish %s' % i)
                # if( _loss <= -120.):
                #     self.writer(_image_path,write_image)
                #     print('finish one attack  weight: %s with step: %s' %  (_weight,i))
                #     self.sess.run(self.tf_assign_init())
                #     break 
                if( i == 500):
                    self.writer(_image_path,write_image)
                    hard_writer.write(_image_path + '\n')
                    print('hard one attack  weight: %s' %  _weight)
                    self.sess.run(self.tf_assign_init())

                # if(i % 10 == 0):
                #     # _inception_res_loss,_inception_res_stop_t,_inception_res_stop_l,_resnet_tel_loss
                #     _, _total_loss,_weight,_stop,_inception_v4_stop_mix,_inception_res_stop_mix,_inception_tel_stop_mix = self.sess.run([
                #         train_op,
                #         self.total_loss,
                #         self.loss_weight,
                #         self.mix_stop,
                #         self.inception_v4_stop_mix,
                #         self.inception_res_stop_mix,
                #         self.inception_tel_stop_mix,
                #         # self.inception_res_loss,
                #         # self.inception_res_stop_t,
                #         # self.inception_res_stop_l,
                #         # self.resnet_tel_loss
                #         ],feed_dict = feed_dict)
                #     print('stop value: %s' % _stop),
                #     print('total_loss: %s' % _total_loss),
                #     print('weight_fit: %s' % _weight)
                #     print('v4_tar: %s' % _inception_v4_stop_mix)
                #     print('v4_tar: %s' % _inception_res_stop_mix)
                #     print('v4_tar: %s' % _inception_tel_stop_mix)

                    # print('v_res_tar: %s' % _inception_res_loss)
                    # print('v4_res_t: %s' % _inception_res_stop_t)
                    # print('v4_res_l: %s' % _inception_res_stop_l)
                    # print('tel_tar: %s' % _resnet_tel_loss)
                    # _, _total_loss,_weight,_stop,_target_cross_entropy_inception_v4,_label_cross_entropy_inception_v4,_target_cross_entropy_inception_v3,_label_cross_entropy_inception_v3,_target_cross_entropy_inception_res,_label_cross_entropy_inception_res,_target_cross_entropy_resnet_50,_label_cross_entropy_resnet_50,_target_cross_entropy_resnet_101,_label_cross_entropy_resnet_101,_target_cross_entropy_resnet_152,_label_cross_entropy_resnet_152 = self.sess.run([
                    #     train_op,
                    #     self.total_loss,
                    #     self.loss_weight,
                    #     self.stop_value,
                    #     self.target_cross_entropy_inception_v4,
                    #     self.label_cross_entropy_inception_v4,
                    #     self.target_cross_entropy_inception_v3,
                    #     self.label_cross_entropy_inception_v3,
                    #     self.target_cross_entropy_inception_res,
                    #     self.label_cross_entropy_inception_res,
                    #     self.target_cross_entropy_resnet_50,
                    #     self.label_cross_entropy_resnet_50,
                    #     self.target_cross_entropy_resnet_101,
                    #     self.label_cross_entropy_resnet_101,
                    #     self.target_cross_entropy_resnet_152,
                    #     self.label_cross_entropy_resnet_152
                    #     ],feed_dict = feed_dict)
                    # print('stop value: %s' % _stop),
                    # print('total_loss: %s' % _total_loss),
                    # print('weight_fit: %s' % _weight)
                    # print('v4_tar: %s' % _target_cross_entropy_inception_v4)
                    # print('v4_lab: %s' % _label_cross_entropy_inception_v4)
                    # print('v3_tar: %s' % _target_cross_entropy_inception_v3)
                    # print('v3_lab: %s' % _label_cross_entropy_inception_v3)
                    # print('v_res_tar: %s' % _target_cross_entropy_inception_res)
                    # print('v_res_lab: %s' % _label_cross_entropy_inception_res)
                    # print('res50_tar: %s' % _target_cross_entropy_resnet_50)
                    # print('res50_lab: %s' % _label_cross_entropy_resnet_50)
                    # print('res101_tar: %s' % _target_cross_entropy_resnet_101)
                    # print('res101_lab: %s' % _label_cross_entropy_resnet_101)
                    # print('res152_tar: %s' % _target_cross_entropy_resnet_152)
                    # print('res152_lab: %s' % _label_cross_entropy_resnet_152)
 
    def train(self):
        self.data.load_fineune_data()
        data_generator = self.data.get_fineune_generator()
        self.data.shuffle()

        label = self.label_label
        logit = self.model.logits
        # logit = self.model()

        # average_grads = self.average_gradients(grads_mix)
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = label,logits = logit))
        self.summary.append(tf.summary.scalar('loss',self.loss))
        grads = self.optimizer.compute_gradients(self.loss,var_list = self.variables_to_train)
        apply_gradient_op = self.optimizer.apply_gradients(grads,global_step = self.global_step)
        train_op = tf.group(apply_gradient_op)

        for var in self.variables_to_train:
            print(var.op.name)

        ## restore and init
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.pre_model)

        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summary)

        step = 0
        for i in range(self.epoch):
            print('start epoch %s' % i)
            while True:
                try:
                    image_content, label_content = next(data_generator)
                    _,summary_str,_label,_logit = self.sess.run([train_op,summary_op,label,logit],feed_dict = {self.input_images:image_content,self.label_label:label_content})
                    step += 1
                    if(step % 10 == 0):
                        summary_writer.add_summary(summary_str,step)
                    if(step % 100 == 0):
                        self.saver_store.save(self.sess,os.path.join(self.save_model,'%s.ckpt' % (self.model_type)))
                except StopIteration:
                    print( 'finish epoch %s' % i )
                    data_generator = self.data.get_fineune_generator()
                    self.data.shuffle()
                    break  

    def eval(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.pre_model)
        print(self.pre_model)
        logit = self.model.logits
        softmax = tf.nn.softmax(logit)
        right = []
        for _ in tqdm(range(1216)):
            _p,content,label,target = next(self.attack_generator)
            target_index = int(label) - 1
            image_content = np.expand_dims(content,0)
            _softmax = self.sess.run(softmax,feed_dict = {self.input_images:image_content})
            pred = np.argsort(np.squeeze(_softmax))[-1]
            print(target_index)
            print(pred)
            print('-------------')
            if(pred == target_index):
                print(pred)
                right.append(_p)
        # v3_old :1002 / 1216
        # v3_new : 1153 / 1216
        # v4_old :1129 / 1216
        # v4_new : 1161 / 1216
        # v_res_old :1184 / 1216
        # v_res_new : / 1216
        print(len(right))
    
    
             

    def eval_local(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.pre_model)

        
        # logit = self.model.logits
        # softmax = tf.nn.softmax(logit)
        image_list =  [ os.path.join('data/test_local',path) for path in os.listdir('data/test_local') ]
        while True:
            for item in image_list:
                content = self.data.rbg2float(cv2.imread(item))
                # resize_size = tuple(np.random.randint(200,400,(2)))
                content = cv2.resize(content,resize_size)
                image_content = np.expand_dims(content,0).astype(np.float32)
                logit = self.model.inception_res(image_content)
                softmax = tf.nn.softmax(logit)
                _softmax = self.sess.run(softmax)
                print(np.sort(np.squeeze(_softmax))[-5:])
                print(np.argsort(np.squeeze(_softmax))[-5:])



        



        

