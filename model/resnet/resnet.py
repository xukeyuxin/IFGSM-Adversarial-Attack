import tensorflow as tf
from model.resnet.resnet_v2 import resnet_arg_scope, resnet_v2_50,resnet_v2_101,resnet_v2_152
slim = tf.contrib.slim

class resnet(object):
    def __init__(self,image,is_training = True):
        self.num_classes = 1000 
        self.X = image
        self.is_training = is_training
        # self.checkpoint_exclude_scopes = "Logits_out"
    def resnet_50(self):
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net, end_points = resnet_v2_50(self.X, is_training=self.is_training)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('resnet_50'):
                # net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
                # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
                # net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
                # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
                net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
                net = tf.squeeze(net,[1,2], name='SpatialSqueeze')
        self.logits = net
        return net

    def resnet_152(self):
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net, end_points = resnet_v2_152(self.X, is_training=self.is_training)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('resnet_152'):
                # net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
                # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
                # net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
                # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
                net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
                net = tf.squeeze(net,[1,2], name='SpatialSqueeze')
        self.logits = net
        return net

    def resnet_101(self):
        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            net, end_points = resnet_v2_101(self.X, is_training=self.is_training)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('resnet_101'):
                # net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out0')
                # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out0')
                # net = slim.conv2d(net, 200, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out1')
                # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b_out1')
                net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Logits_out2')
                net = tf.squeeze(net,[1,2], name='SpatialSqueeze')
        self.logits = net
        return net           

    def g_parameter(self):
        exclusions = ['resnet_50/Logits_out','resnet_101/Logits_out','resnet_152/Logits_out']
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

    def get_train_restore_vars(self):
        if(self.is_training):
            variables_to_restore,variables_to_train = self.g_parameter()
        else:
            variables_to_restore, variables_to_train = slim.get_model_variables(), []
        return variables_to_restore, variables_to_train

