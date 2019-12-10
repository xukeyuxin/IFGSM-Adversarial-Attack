from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from model.inception.inception_v4 import  inception_v4_arg_scope, inception_v4
from model.inception.inception_v3 import  inception_v3_arg_scope, inception_v3
from model.inception.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from util import *
slim = tf.contrib.slim


     
class inception(object):
    def __init__(self,image,is_training = True):
        self.num_classes = 1000 
        self.X = image
        self.is_training = is_training
        self.checkpoint_exclude_scopes = "Logits_out"
    def inception_v4(self, input_image = None, dropout_keep_prob=0.8, is_train=False):
        arg_scope = inception_v4_arg_scope()
        if(input_image is not None):
            input = input_image
        else:
            input = self.X

        with slim.arg_scope(arg_scope):
            net, end_points = inception_v4(input, is_training=self.is_training,reuse=tf.AUTO_REUSE)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('InceptionV4',reuse=tf.AUTO_REUSE):
                # 8 x 8 x 1536
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                        scope='AvgPool_1a_out')
                # 1 x 1 x 1536
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
                net = slim.flatten(net, scope='PreLogitsFlatten_out')
                # 1536
                # net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, self.num_classes, activation_fn=None,scope='Logits_out0')
        self.logits = net
        return net
    
    def inception_v3(self, input_image = None, dropout_keep_prob=0.8, is_train=False):
        arg_scope = inception_v3_arg_scope()
        if(input_image is not None):
            input = input_image
        else:
            input = self.X
        with slim.arg_scope(arg_scope):
            net, end_points = inception_v3(input, is_training=self.is_training,reuse=tf.AUTO_REUSE)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('InceptionV3',reuse=tf.AUTO_REUSE):
                # 8 x 8 x 2048
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                        scope='AvgPool_1a_out')
                # 1 x 1 x 2048
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
                net = slim.flatten(net, scope='PreLogitsFlatten_out')
                # 2048
                # net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, self.num_classes, activation_fn=None,scope='Logits_out0')
        self.logits = net
        return net
    
    def inception_res(self, input_image =None, dropout_keep_prob=0.8, is_train=False):
        arg_scope = inception_resnet_v2_arg_scope()
        if(input_image is not None):
            input = input_image
        else:
            input = self.X

        with slim.arg_scope(arg_scope):
            net, end_points = inception_resnet_v2(input, is_training=self.is_training,reuse=tf.AUTO_REUSE)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('InceptionResnetV2',reuse=tf.AUTO_REUSE):
                # 8 x 8 x 2080
                self.pool_feature = net
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                        scope='AvgPool_1a_out')
                # 1 x 1 x 2080
                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out')
                net = slim.flatten(net, scope='PreLogitsFlatten_out')
                # 2080
                # net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu, scope='Logits_out0')
                net = slim.fully_connected(net, self.num_classes, activation_fn=None,scope='Logits_out0')
        self.logits = net
        return net      
          
    def g_parameter(self):
        exclusions = ['InceptionV4/Logits_out','InceptionResnetV2/Logits_out','InceptionV3/Logits_out']
        # 需要加载的参数。
        variables_to_restore = []
        # 需要训练的参数
        variables_to_train = []
        for var in slim.get_model_variables():
        # 切记不要用下边这个，这是个天大的bug，调试了3天
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    variables_to_train.append(var)
                    print ("ok")
                    print (var.op.name)
                    break
            if not excluded:
                variables_to_restore.append(var)
        return variables_to_restore,variables_to_train

    def get_train_restore_vars(self,scope_list = None):
        if(self.is_training):
            variables_to_restore,variables_to_train = self.g_parameter()
        else:
            variables_to_train = []
            if(scope_list):
                variables_to_restore = slim.get_model_variables(scope = scope_list)
            else:
                variables_to_restore = slim.get_model_variables()
        return variables_to_restore, variables_to_train
