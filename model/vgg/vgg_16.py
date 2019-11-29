import tensorflow as tf
from model.vgg.vgg import vgg_arg_scope, vgg_16
slim = tf.contrib.slim

class VGG16(object):
    def __init__(self,image,is_training = True):
        self.num_classes = 1000 
        self.X = image
        self.is_training = is_training
        self.checkpoint_exclude_scopes = "Logits_out"

    def __call__(self):
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            net, end_points = vgg_16(self.X, is_training=self.is_training)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Logits_out'):
                net = tf.reduce_mean(net, [1,2], keep_dims=True)
                net = slim.conv2d(net, self.num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
                net = tf.squeeze(net,[1,2], name='fc8/squeezed')
        self.logits = net
        return net
    def g_parameter(self):
        exclusions = []
        if self.checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in self.checkpoint_exclude_scopes.split(',')]
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

