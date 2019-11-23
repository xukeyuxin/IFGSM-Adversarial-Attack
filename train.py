import tensorflow as tf
from model.resnet import ResNet
import numpy as np
import json
import pickle 
from op_base import op_base
import os

class Classify(op_base):
    def __init__(self,sess,args):
        op_base.__init__(self,args)
        self.sess = sess
        self.summary = []
        self.model = ResNet(is_training = False)
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
        self.global_step = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),dtype= tf.int64,
                            trainable=False)
        lr = tf.train.exponential_decay(
            self.lr,
            self.global_step,
            self.lr_decay_step,
            self.lr_decay_factor,
            staircase=True)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.opt_momentum)
        self.input_images = tf.placeholder(tf.float32,shape = [None,self.image_height,self.image_weight,3])
        self.input_label = tf.placeholder(tf.float32,shape = [None,self.class_nums])

        # self.feat = self.model(self.input_images)


    def graph(self,logit):
        loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logit,labels = self.input_label))
        l2_regularization = self.weight_decay * tf.add_n( [ tf.nn.l2_loss(v) for v in tf.trainable_variables() ] )
        loss = loss_softmax + l2_regularization
        self.summary.append(tf.summary.scalar('loss_softmax',loss_softmax))
        self.summary.append(tf.summary.scalar('l2_regularization',l2_regularization))
        grads = self.optimizer.compute_gradients(loss,self.get_vars('logits'))
        return grads

    def eval_label(self):
        task_num = 1216
        total_index = 1
        f = open('attack_label.pickle','ab+')
        while True:
            try:
                _image_content,_label,_target = next(self.attack_generator)
                simliar_value = 0.

                _image_content = np.expand_dims(_image_content,axis = 0)
                _item_label_feat = self.sess.run(self.feat,feed_dict = {self.input_images:_image_content})
                label_dims = np.squeeze(_item_label_feat).argsort()[-self.choose_dims:]

                _target_generator = self.target_generator(_label)
                final_target_feat = np.zeros((2048))

                index = 0
                while True:  
                    target_content, target_path = next(_target_generator)
                    target_content = np.expand_dims(target_content,axis = 0)
                    _item_target_feat = self.sess.run(self.feat,feed_dict = {self.input_images:target_content})
                    final_target_feat += np.squeeze(_item_target_feat)
                    index += 1
                    if(index == 100):
                        target_dims = final_target_feat.argsort()[-self.choose_dims:]
                        break

                _item = (_label,label_dims,target_dims)
                pickle.dump(_item,f)
                print('analy finish %s / %s' % (total_index,task_num))
                total_index += 1

            except StopIteration:
                print('finish all')


        # print(final_target_feat.sort())
        # print(final_target_feat.argsort()[-10:])
        


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
        with tf.device('cpu:0'):
            self.data.load_fineune_data()
            data_generator = self.data.get_fineune_generator()
            grads_mix = []


            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpu):
                    with tf.device('gpu:%s' % i):
                        with tf.name_scope('gpu_%s' % i) as scope:
                            logit = self.model(self.input_images,get_feature = False)
                            grads = self.graph(logit)

                            tf.get_variable_scope().reuse_variables()
                            grads_mix.append(grads)

            average_grads = self.average_gradients(grads_mix)
            apply_gradient_op = self.optimizer.apply_gradients(average_grads, global_step=self.global_step)

            ## restore and init
            self.saver = tf.train.Saver(tf.global_variables(),max_to_keep = 5)
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
                        _,summary_str = self.sess.run([apply_gradient_op,summary_op],feed_dict = {self.input_images:image_content,self.input_label:label_content})
                        step += 1
                        if(step % 10 == 0):
                            summary_writer.add_summary(summary_str,step)
                        if(step % 100 == 0):
                            self.saver.save(self.sess,os.path.join(self.save_model,'checkpoint_%s_%s.ckpt' % (i,step)))
                    except StopIteration:
                        print( 'finish epoch %s' % i )
                        self.data.shuffle()
                        continue  




        

