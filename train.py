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
        self.mask = tf.placeholder(tf.float32,shape = [1,self.image_height,self.image_weight,3])
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
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep = 5)

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

    def eval_finetune(self):

        logit = self.model.logit
        prob_topk, pred_topk = tf.nn.top_k(logit, k=5)

        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.pre_model)
        
        short_queue_content = []
        short_queue_label = []
        
        root_dir = 'data/ISLVRC2012/n01440764'
        image_test = os.listdir(root_dir)
        for item in np.random.choice(image_test,1):
            # _image_content,_label,_target = next(self.attack_generator)
            short_queue_content.append(self.data.rbg2float(cv2.imread(os.path.join(root_dir,item))))
            
        _image_content = np.asarray(short_queue_content)
        pred, _prob_topk, _pred_topk = self.sess.run([logit,prob_topk, pred_topk],feed_dict = {self.input_images:_image_content})

        print(np.squeeze(_prob_topk), np.squeeze(_pred_topk))

        # .argsort()[-self.choose_dims:]
        # print(np.squeeze(pred).argsort(a))
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
        result = tf.truncated_normal(shape,mean = 0, stddev = dev)
        return result

    def init_noise(self):
        self.noise = self.xavier_initializer([1,299,299,3])
        # tf.get_variable('noise', [self.image_height, self.image_weight,3], tf.float32, xavier_initializer())

    def eval_label(self):
        feat = tf.squeeze(self.model.feat)
        logit = tf.nn.softmax(self.model.logit)

        ## restore and init
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.pre_model)
        for index in tqdm(range(1,1001)):
            root_pickle_dir = os.path.join('data/features_class')
            if(not os.path.exists(root_pickle_dir)):
                os.mkdir(root_pickle_dir)
            
            f_a = open(os.path.join(root_pickle_dir,'%s.pickle' % index),'ab+')

            ### target 
            _target_generator = self.target_generator(index)
            while True:  
                try:
                    target_content, target_path = next(_target_generator)
                    target_content = np.expand_dims(target_content,axis = 0)
                    target_feat,target_logit = self.sess.run([feat,logit],feed_dict = {self.input_images:target_content})
                    target_logit = np.squeeze(target_logit)
                    item_tuple = (target_path,target_feat,target_logit)
                    pickle.dump(item_tuple,f_a)
                except StopIteration:
                    print('analy target finish %s / %s' % (index,1000))
                    break
            
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
        return mask * self.gaussian_blur(self.noise) 
    
    def write_noise(self,mask,noise):
        return mask * self.gaussian_blur(noise) 
    
    def float2rgb(self,input):
        return input * 127.5 + 127.5
    
    def resize(self,input):
        return np.reshape(input,(299,299,3))

    def feat_graph(self,combine_images,pre_noise,label_feature,target_feature,old_grad,momentum = 0.25, lr = 1.,index = 0):

        ### 调参
        alpha1 = 1
        alpha2 = 1  
        
        with_noise_feat = self.tensor_normal_2(self.model(combine_images))
        loss_feat_1 = tf.reduce_sum(label_feature * with_noise_feat) 
        loss_feat_2 = tf.reduce_sum(target_feature * with_noise_feat)

        alpha1 = tf.cast(tf.cond(loss_feat_1 < 0.3,lambda: 1.,lambda: 0.), tf.float32)
        alpha2 = tf.cast(tf.cond(loss_feat_1 > 0.7,lambda: 1.,lambda: 0.), tf.float32)

        loss_feat = alpha1 * loss_feat_1 - alpha2 * loss_feat_2
        feat_grad = tf.gradients(ys = loss_feat,xs = self.noise)[0] ## (299,299,3)

        loss1_v = feat_grad * (1 - momentum) + old_grad * momentum
        loss_l2 = tf.sqrt(tf.reduce_sum(pre_noise**2))
        loss_tv = self.tv_loss(pre_noise)

        r3 = 1
        if index > 100:
            r3 *= 0.1
        if index > 200:
            r3 *= 0.1

        loss_weight = r3 * 0.025 * loss_l2 + r3 * 0.004 * loss_tv
        finetune_grad = tf.gradients(loss_weight,self.noise)[0]    

        tmp_noise = self.noise - lr * (finetune_grad + loss1_v)
        tmp_noise = tmp_noise + tf.clip_by_value(self.input_images, -1., 1.) - self.input_images
        tmp_noise = tf.clip_by_value(tmp_noise,-0.25, 0.25)

        self.loss_feat_1 = loss_feat_1
        self.loss_feat_2 = loss_feat_2
        self.loss_weight = loss_weight
        return tmp_noise, loss1_v
    
    def update_op(self,new_noise):
        return tf.assign(self.noise,new_noise)

    def make_feed_dict(self,input_image,label_feature,target_feature,mask):
        return {self.input_images:input_image,self.label_feature:label_feature,self.target_feature:target_feature,self.mask:mask} 


    def attack(self):
        self.label_feature = tf.placeholder(tf.float32,shape = [2048])
        self.target_feature = tf.placeholder(tf.float32,shape = [2048])
    

        ## restore and init
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.pre_model)
        
        root_dir = os.path.join('data/feature')
        attack_tasks = os.listdir(root_dir)
        for item_index in attack_tasks:
            _image_path,_image_content,_label,_target = next(self.attack_generator)
            old_grad = tf.constant(0.)
            with open(os.path.join(root_dir,item_index,'target_mean_feature.pickle'),'rb') as f_t,open(os.path.join(root_dir,item_index,'label_mean_feature.pickle'),'rb') as f_l,open(os.path.join(root_dir,item_index,'label_mask.pickle'),'rb') as f_m:
                target_feature = self.normal_2(pickle.load(f_t)) # (2048,)
                label_feature = self.normal_2(pickle.load(f_l)) # (2048)
                _image_content = np.reshape( _image_content, [1,299,299,3] ) # (1,299,299,3)
                mask = np.ones([1,299,299,1])
                # mask = np.reshape( pickle.load(f_m),[1,299,299,1] ) * 0.   # (1,299,299,3)
                print('start attack %s' % _image_path)
                for i in tqdm(range(1,1000)):
                    pre_noise = self.pre_noise(mask)
                    combine_images = _image_content + pre_noise
                    print(combine_images.shape)

                    self.noise,old_grad = self.feat_graph(combine_images,pre_noise,label_feature,target_feature,old_grad,index = i)
                    
                    # update_noise, _old_grad,feat_1,feat_2,weight = self.feat_graph(combine_images,pre_noise,label_feature,target_feature,old_grad)
                    # _noise,old_grad,_feat_1,_feat_2,_weight = self.sess.run([update_noise, _old_grad,feat_1,feat_2,weight],feed_dict = {self.input_images:combine_images.eval()})
                    # print('feat_label: %s' % _feat_1)
                    # print('feat_target: %s' % _feat_2)
                    # print('weight_fit: %s' % _weight)
                    # self.noise = tf.convert_to_tensor(_noise)

                    if(i % 25 == 0):
                        _noise,_feat_1,_feat_2,_weight = self.sess.run([self.noise,self.loss_feat_1,self.loss_feat_2 ,self.loss_weight],feed_dict = {self.input_images:_image_content})
                        print('feat_label: %s' % _feat_1)
                        print('feat_target: %s' % _feat_2)
                        print('weight_fit: %s' % _weight)
                        write_noise = self.sess.run(self.write_noise(mask,_noise))
                        new_content = self.resize(self.float2rgb(np.clip(write_noise + _image_content,-1,1)))
                        noise_image = self.resize(self.float2rgb(write_noise))
                        image_combine_with_noise = os.path.join('data','result',_image_path)
                        noise_image_path = os.path.join('data','result','noise.png')
                        cv2.imwrite(image_combine_with_noise,new_content)
                        cv2.imwrite(noise_image_path,noise_image)

                print('finish %s' % _image_path)




    def eval(self):
        _image_content,_label,_target = next(self.attack_generator)
        simliar_value = 0.
        _target_generator = self.target_generator(_target)
        while True:
            target_mix = []
            target_path_mix = []
            for i in range(self.batch_size):
                try:
                    target_content, target_path = next(_target_generator)
                    target_mix.append(target_content)
                    target_path_mix.append(target_path)
                except StopIteration:
                    if(target_mix):
                        print('finish add target %s' % _target)
                        break
                    else:
                        print('find simliar_value: %s simliar_path: %s' % (simliar_value,simliar_path))
                        return 
            print('analy new batch')
            target_mix = np.asarray(target_mix)
            label_mix = np.asarray([_image_content])

            _target_feat = self.sess.run(self.feat,feed_dict = {self.input_images:target_mix})
            _label_feat = self.sess.run(self.feat,feed_dict = {self.input_images:label_mix})

            target_feat = np.squeeze(_target_feat)
            label_feat = np.squeeze(_label_feat)

            similar_feat = np.array( [label_feat.reshape((-1)) for i in range(len(target_feat))] ) * target_feat
            simliar_sum = np.sum( similar_feat,axis = (-1) )
            _simliar_value = np.max(simliar_sum)
            if(_simliar_value > simliar_value):
                simliar_index = np.argmax(simliar_sum)
                simliar_path = target_path_mix[simliar_index]
                simliar_value = _simliar_value
                print('find new path: %s' % simliar_path )
        

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




        

