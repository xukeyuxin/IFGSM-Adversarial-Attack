import tensorflow as tf
from model.resnet import ResNet
import numpy as np
import json
import pickle 
from op_base import op_base

class Classify(op_base):
    def __init__(self,sess,args):
        op_base.__init__(self,args)
        self.sess = sess
        self.model = ResNet(is_training = True)
        self.init_model()
        self.attack_generator = self.data.load_attack_image()
        self.target_generator = self.data.load_ImageNet_target_image
    def convert(self,input):
        return tf.convert_to_tensor(input)
    def init_model(self):
        self.input_images = tf.placeholder(tf.float32,shape = [None,self.image_height,self.image_weight,3])
        self.feat = self.model(self.input_images)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, self.pre_model)
    
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


        

