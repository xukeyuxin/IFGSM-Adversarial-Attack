import tensorflow as tf
from model.resnet import ResNet
import numpy as np
import json
import pickle 
from op_base import op_base
import os
import cv2
from tqdm import tqdm
class Classify(op_base):
    def __init__(self,sess,args):
        op_base.__init__(self,args)
        self.sess = sess
        self.summary = []
        self.input_images = tf.placeholder(tf.float32,shape = [None,self.image_height,self.image_weight,3])
        self.model = ResNet(self.input_images, is_training = False)
        
        self.init_model()
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
    def init_model(self):
        self.model()
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
    def normal_2(input):
        return  input / ( np.sqrt( np.sum(np.square(input) ) ) )
    # def eval_label(self):
    #     task_num = 1216
    #     total_index = 1
    #     f_f = open('attack_sim_feat.pickle','ab+')
    #     f_l = open('attack_sim_feat.pickle','ab+')
    #     while True:
    #         try:
    #             _image_name,_image_content,_label,_target = next(self.attack_generator)
    #             simliar_value = 0.

    #             _image_content = np.expand_dims(_image_content,axis = 0)
    #             feat = tf.squeeze(self.model.feat)
    #             logit = tf.nn.softmax(self.model.logit)
    #             _label_feat,_label_logit = self.sess.run([feat,logit],feed_dict = {self.input_images:_image_content})

    #             _label_feat_normal = self.normal_2(_label_feat)

    #             _target_generator = self.target_generator(_target)

    #             best_feat_value = 0.
    #             best_logit_arg = np.argmax()
    #             best_pred = np.zeros((1000))
    #             while True:  
    #                 target_content, target_path = next(_target_generator)
    #                 target_content = np.expand_dims(target_content,axis = 0)
    #                 _target_feat,_target_logit = self.sess.run([feat,logit],feed_dict = {self.input_images:_image_content})

    #                 cos_dis = _label_feat_normal * self.normal_2(_target_feat)
    #                 if(cos_dis > best_feat_value):
    #                     choose_feat_target = target_path
    #                     best_feat_value = cos_dis



    #             _item = (_label,_label_feat,_label_logit)
    #             pickle.dump(_item,f)
    #             print('analy finish %s / %s' % (total_index,task_num))
    #             total_index += 1

    #         except StopIteration:
    #             print('finish all')


    
    def eval_new(self):
        task_num = 1216
        index = 1
        feat = tf.squeeze(self.model.feat)
        logit = tf.nn.softmax(self.model.logit)

        ## restore and init
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.pre_model)
        while True:
            try:
                root_pickle_dir = os.path.join('data/features',str(index))
                if(not os.path.exists(root_pickle_dir)):
                    os.mkdir(root_pickle_dir)
                
                f_a = open(os.path.join(root_pickle_dir,'single_label.pickle'),'ab+')
                f_l = open(os.path.join(root_pickle_dir,'label_feature_logit.pickle'),'ab+')
                f_t = open(os.path.join(root_pickle_dir,'target_feature_logit.pickle'),'ab+')

                _image_name,_image_content,_label,_target = next(self.attack_generator)
                _image_content = np.expand_dims(_image_content,axis = 0)
                attack_label_feat,attack_label_logit = self.sess.run([feat,logit],feed_dict = {self.input_images:_image_content})
                attack_label_logit = np.squeeze(attack_label_logit)
                attack_tuple = (_image_name,attack_label_feat,attack_label_logit)
                pickle.dump(attack_tuple,f_a)

                ### label 
                _label_generator = self.target_generator(_label)
                while True:  
                    try:
                        label_content, label_path = next(_label_generator)
                        label_content = np.expand_dims(label_content,axis = 0)
                        label_feat,label_logit = self.sess.run([feat,logit],feed_dict = {self.input_images:_image_content})
                        label_logit = np.squeeze(label_logit)
                        
                        item_tuple = (label_path,label_feat,label_logit)
                        pickle.dump(item_tuple,f_l)
                    except StopIteration:
                        print('analy label finish %s / %s' % (index,task_num))
                        break
                    

                ### target 
                _target_generator = self.target_generator(_target)
                while True:  
                    try:
                        target_content, target_path = next(_target_generator)
                        target_content = np.expand_dims(target_content,axis = 0)
                        target_feat,target_logit = self.sess.run([feat,logit],feed_dict = {self.input_images:target_content})
                        target_logit = np.squeeze(target_logit)
                        item_tuple = (target_path,target_feat,target_logit)
                        pickle.dump(item_tuple,f_t)
                    except StopIteration:
                        print('analy target finish %s / %s' % (index,task_num))
                        break
                
                index += 1

            except StopIteration:
                print('finish all')


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




        

