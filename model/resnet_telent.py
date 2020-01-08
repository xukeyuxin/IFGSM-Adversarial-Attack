"""ResNet model
Related papers:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

"""Global Options
"""
tf.app.flags.DEFINE_string('mode', 'train',
   "run coder in tain or validation mode")
tf.app.flags.DEFINE_integer('max_to_keep', 200,
   "save checkpoint here")

"""Data Options
"""
tf.app.flags.DEFINE_string('data_dir', './data/train/',
   "Path to the data TFRecord of Example protos. Should save in train and val")
tf.app.flags.DEFINE_integer('batch_size', 512,
   "Number of images to process in a batch.")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
   "Number of preprocessing threads per tower. Please make this a multiple of 4")
tf.app.flags.DEFINE_integer('file_shuffle_buffer', 1500,
   "buffer size for file names")
tf.app.flags.DEFINE_integer('shuffle_buffer', 2048,
   "buffer size for samples")
tf.app.flags.DEFINE_boolean('with_bbox', True, 
   "whether use bbox in train set")

"""Model Options
"""
tf.app.flags.DEFINE_integer('class_num', 1000,
  "distinct class number")
tf.app.flags.DEFINE_integer('resnet_size', 101,
  "resnet block layer number [ 18, 34, 50, 101, 152, 200 ]")
tf.app.flags.DEFINE_string('data_format', 'channels_last',
  "data format for the input and output data [ channels_first | channels_last ]")
tf.app.flags.DEFINE_integer('image_size', 224,
   "default image size for model input layer")
tf.app.flags.DEFINE_integer('image_channels', 3,
   "default image channels for model input layer")
tf.app.flags.DEFINE_float('batch_norm_decay', 0.997,
   "use for batch normal moving avg")
tf.app.flags.DEFINE_float('batch_norm_epsilon', 1e-5,
   "use for batch normal layer, for avoid divide by zero")
tf.app.flags.DEFINE_float('mask_thres', 0.7,
   "mask thres for balance pos neg")
tf.app.flags.DEFINE_float('neg_select', 0.3,
   "how many class within only negtive samples in a batch select to learn")

"""Train Options
"""
tf.app.flags.DEFINE_boolean('restore', False,
   "whether to restore weights from pretrained checkpoint.")
tf.app.flags.DEFINE_integer('num_gpus', 1,
   "How many GPUs to use.")
tf.app.flags.DEFINE_string('optimizer','mom',
   "optimation algorthm")
tf.app.flags.DEFINE_float('opt_momentum', 0.9,
   "moment during learing")
tf.app.flags.DEFINE_float('lr', 0.1,
   "Initial learning rate.")
tf.app.flags.DEFINE_integer('lr_decay_step', 0,
   "Iterations after which learning rate decays.")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1,
   "Learning rate decay factor.")
tf.app.flags.DEFINE_float('weight_decay', 0.0001,
   "Tainable Weight l2 loss factor.")
tf.app.flags.DEFINE_integer('warmup', 0,
   "Steps when stop warmup, need when use distributed learning")
tf.app.flags.DEFINE_float('lr_warmup', 0.1,
   "Initial warmup learning rate, need when use distributed learning")
tf.app.flags.DEFINE_integer('lr_warmup_decay_step', 0,
   "Iterations after which learning rate decays, need when use distributed learning")
tf.app.flags.DEFINE_float('lr_warmup_decay_factor', 1.414,
   "Warmup learning rate decay factor, need when use distributed learning")
tf.app.flags.DEFINE_integer('max_iter', 1000000,
   "max iter number for stopping.-1 forever")
tf.app.flags.DEFINE_integer('test_interval', 0,
   "iterations interval for evluate model")
tf.app.flags.DEFINE_integer('test_iter', 0,
   "iterations for evluate model")
tf.app.flags.DEFINE_integer('prof_interval', 10,
   "iterations for print training time cost")
tf.app.flags.DEFINE_integer('log_interval', 0,
  "iterations for print summery log")
tf.app.flags.DEFINE_string('log_dir', './out/log/',
   "Directory where to write event logs")
tf.app.flags.DEFINE_string('model_dir', './out/checkpoint/',
   "path for saving learned tf model")
tf.app.flags.DEFINE_string('tmp_model_dir', './out/tmp/checkpoint/',
   "The directory where the temporary model will be stored")
tf.app.flags.DEFINE_integer('snapshot', 0,
   "Iteration for saving model snapshot")
tf.app.flags.DEFINE_integer('epoch_iter', 0,
   "Iteration for epoch ")
tf.app.flags.DEFINE_float('drop_rate', 0.5, 
   "DropOut rate")
tf.app.flags.DEFINE_integer('random_seed', 1234,
   "Random sedd for neigitive class selected")
tf.app.flags.DEFINE_string('pretrain_ckpt', '',
   'pretrain checkpoint file')
tf.app.flags.DEFINE_boolean('FixBlock2', False,
   'whether to fix the first two block, used for fintuning')


"""eval options
"""
tf.app.flags.DEFINE_integer('visiable_gpu', 0,
   "wihch gpu can use")
tf.app.flags.DEFINE_string('piclist', '',
   "eval picture list")
tf.app.flags.DEFINE_integer('interval', 32,
   "eval chekpoint interval")
tf.app.flags.DEFINE_integer('start', 0,
   "the start index of ckpts")

class ResNet(object):
  def __init__(self, images,is_training):
    """Net constructor
    Args:
      images: 4-D Tensor of images with Shape [batch_size, image_size, image_size, 3]
      is_training: bool, used in batch normalization
    Return:
      A wrapper For building model
    """
    self.is_training = is_training
    self.filters =  [256, 512, 1024, 2048] # feature map size for each stages
    self.strides =  [2,   2,   2,    2]    # conv strides for each stages's first block
    if FLAGS.resnet_size == 50:            # resnet size paramters
      self.stages = [3,   4,   6,    3]
    elif FLAGS.resnet_size == 101:
      self.stages = [3,   4,   23,   3]
    elif FLAGS.resnet_size == 152:
      self.stages = [3,   8,   36,   3]
    else:
      raise ValueError('resnet_size %d Not implement:' % FLAGS.resnet_size)
    self.data_format = FLAGS.data_format
    self.num_classes = FLAGS.class_num
    self.images = images
    
    # if(image_with_noise is not None):
    #   self.image_with_noise = image_with_noise
    # if self.data_format == "NCHW":  
    #   self.images = tf.transpose(images, [0, 3, 1, 2])
  def get_restore_variable(self):
    return [ var for var in tf.global_variables() if var.op.name.startswith('resnet_tel') ]
  def __call__(self,input_image = None,get_feature = True):
    # Initial net
    with tf.variable_scope('resnet_tel',reuse = tf.AUTO_REUSE):
      with tf.variable_scope('init',reuse = tf.AUTO_REUSE):
        if(input_image is not None):
            x = input_image
        else:
            x = self.images
        x = self._pre_padding_conv('init_conv', x, 7, 64, 2)

      # 4 stages 
      for i in range(0, len(self.stages)):
        with tf.variable_scope('stages_%d_block_%d' % (i,0),reuse = tf.AUTO_REUSE):
          x = self._bottleneck_residual(
                x, 
                self.filters[i], 
                self.strides[i], 
                'conv',
                self.is_training)
        for j in range(1, self.stages[i]):
          with tf.variable_scope('stages_%d_block_%d' % (i,j),reuse = tf.AUTO_REUSE):
            x = self._bottleneck_residual(
                  x, 
                  self.filters[i], 
                  1,
                  'identity', 
                  self.is_training)
      
      # class wise avg pool
      with tf.variable_scope('global_pool',reuse = tf.AUTO_REUSE):
        x = self._batch_norm('bn', x, self.is_training)
        x = self._relu(x)
        x = self._global_avg_pool(x)


      # extract features
      self.feat=x
      
      # logits
      with tf.variable_scope("logits",reuse = tf.AUTO_REUSE):
        x = self._fully_connected(x, out_dim=self.num_classes)
        self.logits = x
      return x


  def _bottleneck_residual(self, x, out_channel, strides, _type, is_training = False):
    """Residual Block
     Args:
       x : A 4-D tensor
       out_channels : out feature map size of residual block
       strides : conv strides of block
       _type: short cut type, 'conv' or 'identity'
       is_training :  A Boolean for whether the model is in training or inference mdoel
    """
    # short cut
    orig_x = x
    if _type=='conv':
      orig_x = self._batch_norm('conv1_b1_bn', orig_x, is_training)
      orig_x = self._relu(orig_x)
      orig_x = self._pre_padding_conv('conv1_b1', orig_x, 1, out_channel, strides)

    # bottleneck_residual_block
    x = self._batch_norm('conv1_b2_bn', x, is_training)
    x = self._relu(x)
    x = self._pre_padding_conv('conv1_b2', x, 1, out_channel/4, 1)
    x = self._batch_norm('conv2_b2_bn', x, is_training)
    x = self._relu(x)
    x = self._pre_padding_conv('conv2_b2', x, 3, out_channel/4, strides)
    x = self._batch_norm('conv3_b2_bn', x, is_training)
    x = self._relu(x)
    x = self._pre_padding_conv('conv3_b2', x, 1, out_channel, 1)

    # sum
    return x + orig_x

  def _batch_norm(self, name, x, is_training=False):
    """Batch normalization.
     Considering the performance, we use batch_normalization in contrib/layers/python/layers/layers.py
     instead of tf.nn.batch_normalization and set fused=True
     Args:
       x: input tensor
       is_training: Whether to return the output in training mode or in inference mode, use the argment
                    in finetune
    """
    with tf.variable_scope(name):
      return tf.layers.batch_normalization(
             inputs=x,
             axis=1 if self.data_format == 'NCHW' else 3,
             momentum = FLAGS.batch_norm_decay,
             epsilon = FLAGS.batch_norm_epsilon,
             center=True,
             scale=True,
             trainable=is_training,
             fused=True
             )

  def _pre_padding(self, x, kernel_size):
    """Padding Based On Kernel_size"""
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if self.data_format == 'NCHW':
      x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return x 

  def _pre_padding_conv(self, name, x, kernel_size, out_channels, strides, bias=False):
    """Convolution
    As the way of padding in conv is depended on input size and kernel size, which is very different with caffe
    So we will do pre-padding to Align the padding operation.
     Args:
       x : A 4-D tensor 
       kernel_size : size of kernel, here we just use square conv kernel
       out_channels : out feature map size
       strides : conv stride
       bias : bias may always be false
    """
    if strides > 1:
      x = self._pre_padding(x, kernel_size)
    with tf.variable_scope(name):
      return tf.layers.conv2d(
             inputs = x,
             filters = out_channels,
             kernel_size=kernel_size,
             strides=strides,
             padding=('SAME' if strides == 1 else 'VALID'), 
             use_bias=bias,
             kernel_initializer=tf.variance_scaling_initializer(),
             data_format= 'channels_first' if self.data_format == 'NCHW' else 'channels_last')

  def _relu(self, x, leakiness=0.0):
    """
    Relu. With optical leakiness support
    Note: if leakiness set zero, we will use tf.nn.relu for concern about performance
     Args:
       x : A 4-D tensor
       leakiness : slope when x < 0
    """
    if leakiness==0.0:
      return tf.nn.relu(x)
    else:
      return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _global_avg_pool(self, x):
    """
    Global Average Pool, for concern about performance we use tf.reduce_mean 
    instead of tf.layers.average_pooling2d
     Args:
       x: 4-D Tensor
    """
    assert x.get_shape().ndims == 4
    axes = [2, 3] if self.data_format == 'NCHW' else [1, 2]
    return tf.reduce_mean(x, axes, keep_dims=True)

  def _fully_connected(self, x, out_dim):
    """
    As tf.layers.dense need 2-D tensor, reshape it first
    Args:
      x : 4-D Tensor
      out_dim : dimensionality of the output space.
    """
    assert x.get_shape().ndims == 4
    axes = 1 if self.data_format == 'NCHW' else -1
    x = tf.reshape(x, shape=[-1, x.get_shape()[axes]])
    return tf.layers.dense(x, units = out_dim)
