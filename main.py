#coding:utf-8
import tensorflow as tf
from train import Classify
import argparse
import os
import sys


parser = argparse.ArgumentParser()

# Train Data
# parser.add_argument("-dd", "--data_dir", type=str, default="./data")
parser.add_argument("-dd", "--train_image_path", type=str, default="data/ISLVRC2012")
parser.add_argument("-ic", "--image2class", type=str, default="data/image2class")
parser.add_argument("-ci", "--classFile", type=str, default="data/class.txt")
parser.add_argument("-sd", "--summary_dir", type=str, default="logs")
parser.add_argument("-pm", "--pre_model", type=str, default="model/ckpt-resnet101-mlimages-imagenet/resnet.ckpt")
parser.add_argument("-sm", "--save_model", type=str, default="model/finetune_model")
parser.add_argument("-mp", "--attack_image", type=str, default="data/dev.csv")
parser.add_argument("-atc", "--attack_content", type=str, default="data/images")
parser.add_argument("-otin", "--one_target_image_num", type=int, default=20)



# Train Iteration
parser.add_argument("-iw", "--image_weight", type=int, default=299)
parser.add_argument("-ih", "--image_height", type=int, default=299)
parser.add_argument("-cd", "--choose_dims", type=int, default=10)
parser.add_argument("-mt", "--max_var", type=int, default=32)
parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-cs", "--change_size", type=int, default=4)
parser.add_argument("-tu", "--train_utils", type=str, default='gpu')
parser.add_argument("-l", "--lr", type=float, default=1e-1)
parser.add_argument("-ldt", "--lr_decay_step", type=int, default=0)
parser.add_argument("-ldf", "--lr_decay_factor", type=float, default=0.1)
parser.add_argument("-om", "--opt_momentum", type=float, default=0.9)
parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
parser.add_argument("-ng", "--num_gpu", type=int, default=1)





parser.add_argument("-ac", "--action", type=str, default='train')



args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num


dir_names = ['eval','logs','model','data']
for dir in dir_names:
    if(not os.path.exists(dir)):
        os.mkdir(dir)

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        model = Classify(sess,args)
        model.train()
        # model.eval_label()
        # if(args.action == 'train'):
        #     model.train(is_training = True)
        # elif(args.action == 'test'):
        #     model.eval()
