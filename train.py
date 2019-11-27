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
        self.input_GBR_images = tf.placeholder(tf.float32,shape = [1,self.image_height,self.image_weight,3])
        self.target_feature = tf.placeholder(tf.float32,shape = [2048])
        self.target_label = tf.placeholder(tf.int32,shape = [1,1000])
        self.label_feature = tf.placeholder(tf.float32,shape = [2048])
        self.label_label = tf.placeholder(tf.int32,shape = [1,1000])
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
        input_GBR_images = np.expand_dims(np.squeeze(input_image)[...,::-1],0)
        return {self.input_images:input_image,self.input_GBR_images:input_GBR_images,self.target_label:target_label,self.label_label:label_label,self.mask:mask,self.index:index} 
        # return {self.input_images:input_image,self.target_feature:target_feature,self.label_feature:label_feature,self.mask:mask,self.index:index} 
       
    def init_noise(self):
        ## init
        tmp_noise_init = self.xavier_initializer([1,299,299,3])
        grad_init = 0.
        self.tmp_noise = tf.get_variable('noise',shape = [1,299,299,3], initializer= tf.constant_initializer(tmp_noise_init))
        self.v1_grad = tf.get_variable('noise_grad',shape = [1,299,299,3],initializer= tf.constant_initializer(grad_init))
    


    def attack_graph(self,lr = 1.,momentum = 0.):
        def entropy_loss(label,logits,epsilon = 1e-1):
            _label = tf.cast(label,tf.float32)
            _logits = tf.squeeze(logits)
            return  - tf.reduce_sum(_label * tf.log(logits + epsilon) , axis = 1)
        tmp_noise = self.pre_noise(self.mask)
        self.combine_images = tf.clip_by_value(self.input_images + tmp_noise,-1.,1.)
        self.combine_images_GBR = tf.clip_by_value(self.input_GBR_images + tmp_noise,-1.,1.)

        ### 调参
        alpha1 = 1
        alpha2 = 1  
        
        logits = self.model(self.combine_images)
        logits_GBR = self.model(self.combine_images_GBR)
        self.test_info = tf.nn.softmax(logits)

        label_loss_cross_entropy_rgb = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logits))
        target_loss_cross_entropy_rgb = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logits))

        alpha1 = tf.cast(tf.cond(target_loss_cross_entropy_rgb < 1.,lambda: 0.1,lambda: 1.), tf.float32)
        alpha2 = tf.cast(tf.cond(target_loss_cross_entropy_rgb > 100.,lambda: 0.1,lambda: 1.), tf.float32)

        loss_rbg = alpha1 * target_loss_cross_entropy_rgb - alpha2 * label_loss_cross_entropy_rgb 

        # label_loss_cross_entropy_rgb = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label_label,logits = logits_GBR))
        # target_loss_cross_entropy_rgb = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_label,logits = logits_GBR))
        # loss_bgr = target_loss_cross_entropy_gbr - label_loss_cross_entropy_gbr 

        

        # with_noise_feat = self.tensor_normal_2(self.model(combine_images))
        # loss_feat_1 = tf.reduce_sum(self.label_feature * with_noise_feat) 
        # loss_feat_2 = tf.reduce_sum(self.target_feature * with_noise_feat)

        # alpha1 = tf.cast(tf.cond(loss_feat_1 < 0.3,lambda: 0.1,lambda: 5.), tf.float32)
        # alpha2 = tf.cast(tf.cond(loss_feat_2 > 0.7,lambda: 0.1,lambda: 5.), tf.float32)


        # loss_feat = alpha1 * loss_feat_1 - alpha2 * loss_feat_2

        # feat_grad = tf.gradients(ys = loss_feat,xs = self.noise)[0] ## (299,299,3)
        feat_grad = tf.gradients(ys = loss_rbg,xs = self.tmp_noise)[0] ## (299,299,3)

        loss1_grad = feat_grad * (1 - momentum) + self.v1_grad * momentum
        # loss1_v = feat_grad * (1 - momentum) + old_grad * momentum

        loss_l2 = tf.sqrt(tf.reduce_sum(tmp_noise**2))
        loss_tv = self.tv_loss(tmp_noise)

        r3 = 1.  
        r3 = tf.cond(self.index > 100,lambda: r3 * 0.1,lambda: r3)

        loss_weight = r3 * 0.025 * loss_l2 + r3 * 0.004 * loss_tv   
        finetune_grad = tf.gradients(loss_weight,self.tmp_noise)[0]  
        # finetune_grad = 0.

        # tmp_noise = self.noise - lr * (finetune_grad + loss1_v)
        update_noise = self.tmp_noise - lr * (finetune_grad + loss1_grad)
        update_noise = update_noise + tf.clip_by_value(self.input_images, -1., 1.) - self.input_images
        update_noise = tf.clip_by_value(update_noise,-0.25, 0.25)

        self.label_loss_cross_entropy_rgb = label_loss_cross_entropy_rgb
        self.target_loss_cross_entropy_rgb = target_loss_cross_entropy_rgb
        # self.label_loss_cross_entropy_gbr = label_loss_cross_entropy_gbr
        # self.target_loss_cross_entropy_gbr = target_loss_cross_entropy_gbr

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
            

            label_np = np.array([int(_label)])

            _target = '333'
            target_np = np.array([int(_target)])

            train_op = self.attack_graph()

            with open(os.path.join(root_dir,item_index,'single_label.pickle'),'rb') as f_single, open(os.path.join(root_dir,item_index,'target_best_feature.pickle'),'rb') as f_t:

                # target_feature = self.normal_2(pickle.load(f_t)) # (2048,)
                # label_feature = self.normal_2(pickle.load(f_single)[1]) # (2048)

                label_input = self.data.make_label(label_np)
                target_input = self.data.make_label(target_np)

                _image_content = np.expand_dims(_image_content ,0) # (1,299,299,3)
                mask = np.ones([1,299,299,1])
                print('start attack %s' % _image_path)
                for i in tqdm(range(0,201)):
                    feed_dict = self.make_feed_dict(_image_content,target_input,label_input,mask,i)
                    _ = self.sess.run(train_op,feed_dict = feed_dict)

                    if(i % 50 == 0):
                        _, write_image, label_rgb,target_rgb,_weight = self.sess.run([train_op,self.combine_images,
                        self.label_loss_cross_entropy_rgb,
                        self.target_loss_cross_entropy_rgb,
                        # self.label_loss_cross_entropy_gbr,
                        # self.target_loss_cross_entropy_gbr,
                        self.loss_weight],feed_dict = feed_dict)

                        print('label_rgb: %s' % label_rgb)
                        print('target_rgb: %s' % target_rgb)
                        # print('label_gbr: %s' % label_gbr)
                        # print('target_gbr: %s' % target_gbr)
                        print('weight_fit: %s' % _weight)

                        write_image = self.float2rgb(np.squeeze(write_image))
                        image_combine_with_noise = os.path.join('data','result',_image_path)
                        cv2.imwrite(image_combine_with_noise,write_image)
                         

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




        

