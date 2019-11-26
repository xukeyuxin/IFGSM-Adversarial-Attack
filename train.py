import tensorflow as tf
from model.resnet import ResNet
import numpy as np
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

class Classify(op_base):
    def __init__(self,sess,args):
        op_base.__init__(self,args)
        self.sess = sess
        self.summary = []
        self.input_images = tf.placeholder(tf.float32,shape = [1,self.image_height,self.image_weight,3])
        self.target_feature = tf.placeholder(tf.float32,shape = [2048])
        self.label_feature = tf.placeholder(tf.float32,shape = [2048])
        self.mask = tf.placeholder(tf.float32,shape = [1,self.image_height,self.image_weight,1])
        self.index = tf.placeholder(tf.int32,shape = [])
        self.gaussian_blur = GaussianBlur()

        self.init_noise()
        self.model = ResNet(self.input_images, is_training = False)
        self.model()
        
        self.init_model(noise = True)
        self.attack_generator = self.data.load_attack_image()
        self.target_generator = self.data.load_ImageNet_target_image
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
        # self.model.build_model(noise = noise)

        self.global_step = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),dtype= tf.int64,
                            trainable=False)
        lr = tf.train.exponential_decay(
            self.lr,
            self.global_step,
            self.lr_decay_step,
            self.lr_decay_factor,
            staircase=True)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, momentum=self.opt_momentum)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        

    def graph(self,logit):
        loss_softmax = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logit,labels = self.input_label))
        tf.identity(loss_softmax,name = 'loss_softmax')
        self.summary.append(tf.summary.scalar('loss_softmax',loss_softmax))
        l2_regularization = self.weight_decay * tf.add_n( [ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bn' not in v.name ] )
        l2_regularization_bn = 0.1 * self.weight_decay * tf.add_n(  [ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bn' in v.name ]  )
        loss = loss_softmax + l2_regularization + l2_regularization_bn
        self.summary.append(tf.summary.scalar('l2_regularization',l2_regularization))
        self.summary.append(tf.summary.scalar('l2_regularization_bn',l2_regularization_bn))
        grads = self.optimizer.compute_gradients(loss,tf.trainable_variables())
        return grads

    def normal_2(self,input):
        return  input / ( np.sqrt( np.sum(np.square(input) ) ) )
    
    def tensor_normal_2(self,input):
        return input / tf.sqrt(tf.reduce_sum(tf.square(input)))

    def xavier_initializer(self,shape, factor = 1.):
        factor = float(factor)
        fan_in = shape[-1]
        fan_out = shape[-1]
        variation = (2/( fan_in +fan_out))*factor
        dev = math.sqrt(variation)
        # result = tf.truncated_normal(shape,mean = 0, stddev = dev)
        result = np.random.normal(0,dev,shape)
        return result
       
    def find_sim(self):
        def cal(logits, features):
            logits = np.asarray(logits)
            features = np.asarray(features)
            logits_max = np.max(logits,axis = -1)
            logits_arg = logits_max.argsort()[-100:]
            choose_feature = np.mean(features[logits_arg],axis = 0)
            return choose_feature
        root_dir = os.path.join('data/feature')
        root_mask_dir = os.path.join('data/mask_images')
        class_dir = os.listdir(root_dir)
        for item in class_dir:
            logits_1 = []
            features_1 = []

            logits_2 = []
            features_2 = []

            single = open(os.path.join(root_dir,item,'single_label.pickle'),'rb')
            image_name = pickle.load(single)[0]
            single.close()

            with open(os.path.join(root_dir,item,'target_feature_logit.pickle'),'rb') as f1,open(os.path.join(root_dir,item,'label_feature_logit.pickle'),'rb') as f2, open(os.path.join(root_dir,item,'target_mean_feature.pickle'),'ab+') as f_w, open(os.path.join(root_dir,item,'label_mean_feature.pickle'),'ab+') as f_l,open(os.path.join(root_dir,item,'label_mask.pickle'),'ab+') as f_m:
                while True:
                    try:
                        _path,feature,logit = pickle.load(f1)
                        logits_1.append(logit)
                        features_1.append(feature)
                    except EOFError:
                        print('load finish')
                        break
                while True:
                    try:
                        _path,feature,logit = pickle.load(f2)
                        logits_2.append(logit)
                        features_2.append(feature)
                    except EOFError:
                        print('load finish')
                        break
                
                load_mask = 1. - (cv2.imread(os.path.join(root_mask_dir,image_name),0).astype(np.float32) / 255.)
                load_mask = cv2.threshold(load_mask,0.5,1,cv2.THRESH_BINARY)
                pickle.dump(load_mask,f_m)
                target_feature = cal(logits_1, features_1)
                pickle.dump(target_feature,f_w)

                label_feature = cal(logits_2, features_2)
                pickle.dump(label_feature,f_l)

    
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

    def make_feed_dict(self,input_image,target_feature,label_feature,mask,index):
        return {self.input_images:input_image,self.target_feature:target_feature,self.label_feature:label_feature,self.mask:mask,self.index:index} 
       
    def init_noise(self):
        ## init
        tmp_noise_init = self.xavier_initializer([1,299,299,3])
        grad_init = 0.
        self.tmp_noise = tf.get_variable('noise',shape = [1,299,299,3], initializer= tf.constant_initializer(tmp_noise_init))
        self.v1_grad = tf.get_variable('noise_grad',shape = [1,299,299,3],initializer= tf.constant_initializer(grad_init))

    def attack_graph(self,lr = 1.,momentum = 0.3):
        tmp_noise = self.pre_noise(self.mask)
        combine_images = self.input_images + tmp_noise

        ### 调参
        alpha1 = 1
        alpha2 = 1  
        
        with_noise_feat = self.tensor_normal_2(self.model(combine_images))
        loss_feat_1 = tf.reduce_sum(self.label_feature * with_noise_feat) 
        loss_feat_2 = tf.reduce_sum(self.target_feature * with_noise_feat)

        alpha1 = tf.cast(tf.cond(loss_feat_1 < 0.3,lambda: 1.,lambda: 0.), tf.float32)
        alpha2 = tf.cast(tf.cond(loss_feat_1 > 0.7,lambda: 1.,lambda: 0.), tf.float32)

        loss_feat = alpha1 * loss_feat_1 - alpha2 * loss_feat_2

        # feat_grad = tf.gradients(ys = loss_feat,xs = self.noise)[0] ## (299,299,3)
        feat_grad = tf.gradients(ys = loss_feat,xs = self.tmp_noise)[0] ## (299,299,3)

        loss1_grad = feat_grad * (1 - momentum) + self.v1_grad * momentum
        # loss1_v = feat_grad * (1 - momentum) + old_grad * momentum

        loss_l2 = tf.sqrt(tf.reduce_sum(tmp_noise**2))
        loss_tv = self.tv_loss(tmp_noise)

        r3 = 1.
        r3 = tf.cond(self.index > 100,lambda: r3 * 0.1,lambda: r3)
        r3 = tf.cond(self.index > 200,lambda: r3 * 0.1,lambda: r3)

        loss_weight = r3 * 0.025 * loss_l2 + r3 * 0.004 * loss_tv
        # finetune_grad = tf.gradients(loss_weight,self.noise)[0]    
        finetune_grad = tf.gradients(loss_weight,self.tmp_noise)[0]  

        # tmp_noise = self.noise - lr * (finetune_grad + loss1_v)
        update_noise = self.tmp_noise - lr * (finetune_grad + loss1_grad)
        update_noise = update_noise + tf.clip_by_value(self.input_images, -1., 1.) - self.input_images
        update_noise = tf.clip_by_value(update_noise,-0.25, 0.25)

        self.loss_feat_1 = loss_feat_1
        self.loss_feat_2 = loss_feat_2
        self.loss_weight = loss_weight

        # _noise,_feat_1,_feat_2,_weight = self.sess.run([update_noise,self.loss_feat_1,self.loss_feat_2,self.loss_weight],feed_dict = {self.input_images:_image_content})
        update_value = tf.assign(self.tmp_noise,update_noise)
        update_grad = tf.assign(self.v1_grad,loss1_grad)
        return tf.group(update_value,update_grad)

    def attack(self):


        ## restore and init
        self.sess.run(tf.global_variables_initializer())
        variables_to_restore_image = [v for v in tf.global_variables() if 'noise' not in v.name]
        self.saver = tf.train.Saver(variables_to_restore_image,max_to_keep = 5)
        self.saver.restore(self.sess, self.pre_model)
        
        root_dir = os.path.join('data/feature')
        attack_tasks = os.listdir(root_dir)
        for item_index in attack_tasks:

            _image_path,_image_content,_label,_target = next(self.attack_generator)
            lr = 1
            momentum = 0.3
            train_op = self.attack_graph()

            with open(os.path.join(root_dir,item_index,'target_mean_feature.pickle'),'rb') as f_t,open(os.path.join(root_dir,item_index,'label_mean_feature.pickle'),'rb') as f_l,open(os.path.join(root_dir,item_index,'label_mask.pickle'),'rb') as f_m:
                target_feature = self.normal_2(pickle.load(f_t)) # (2048,)
                label_feature = self.normal_2(pickle.load(f_l)) # (2048)
                _image_content = np.reshape( _image_content, [1,299,299,3] ) # (1,299,299,3)
                mask = np.ones([1,299,299,1])
                print('start attack %s' % _image_path)
                for i in tqdm(range(1,301)):
                    feed_dict = self.make_feed_dict(_image_content,target_feature,label_feature,mask,i)
                    _ = self.sess.run(train_op,feed_dict = feed_dict)

                    if(i % 100 == 0):
                        _, _feat_1,_feat_2,_weight = self.sess.run([train_op,self.loss_feat_1,self.loss_feat_2,self.loss_weight],feed_dict = feed_dict)
                        print('feat_label: %s' % _feat_1)
                        print('feat_target: %s' % _feat_2)
                        print('weight_fit: %s' % _weight)
                        _noise = self.sess.run(self.tmp_noise,feed_dict = feed_dict)
                        write_noise = self.sess.run(self.write_noise(mask,_noise))
                        new_content = self.resize(self.float2rgb(np.clip(write_noise + _image_content,-1,1)))
                        noise_image = self.resize(self.float2rgb(write_noise))
                        image_combine_with_noise = os.path.join('data','result',_image_path)
                        noise_image_path = os.path.join('data','result','noise.png')
                        cv2.imwrite(image_combine_with_noise,new_content)
                        cv2.imwrite(noise_image_path,noise_image)
                         

                print('finish %s' % _image_path)

    def train(self):
        self.data.load_fineune_data()
        data_generator = self.data.get_fineune_generator()
        self.data.shuffle()

        # with tf.variable_scope(tf.get_variable_scope()):
        #     for i in range(self.num_gpu):
        #         with tf.device('gpu:%s' % i):
        #             with tf.name_scope('gpu_%s' % i) as scope:

        logit = self.model(self.input_images,get_feature = False)
        grads = self.graph(logit)

        # average_grads = self.average_gradients(grads_mix)
        apply_gradient_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)
        batchnorm_updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(apply_gradient_op, batchnorm_updates_op)


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
                    _,summary_str = self.sess.run([train_op,summary_op],feed_dict = {self.input_images:image_content,self.input_label:label_content})
                    step += 1
                    if(step % 10 == 0):
                        summary_writer.add_summary(summary_str,step)
                    if(step % 500 == 0):
                        self.saver.save(self.sess,os.path.join(self.save_model,'checkpoint_%s_%s.ckpt' % (i,step)))
                except StopIteration:
                    print( 'finish epoch %s' % i )
                    data_generator = self.data.get_fineune_generator()
                    self.data.shuffle()
                    break  




        

