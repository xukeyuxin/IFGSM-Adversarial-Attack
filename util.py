import tensorflow as tf
import cv2
import numpy as np
import math

def preprocess(image, height, width, bbox):
    # 若没有提供标注框则默认为关注区域为整个图像
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # 转换图像数据类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 随机截取图像减小识别物体大小对模型的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 随机调整图像的大小
    distorted_image = tf.image.resize_images(distorted_image, (height, width), method=np.random.randint(4))
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    # distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image

def tf_resize(input):

    # height,weight =input.get_shape().as_list()[1:3]
    # crop_weight = int(3 / 4 * weight)
    # crop_height = int(3 / 4 * height)
    # new_image = tf.image.random_crop(input,(crop_height,crop_weight))
    new_image = tf.image.random_flip_up_down(input)
    new_image = tf.image.random_flip_left_right(new_image)
    new_image = tf.image.transpose_image(new_image)
    new_image = tf.image.random_brightness(new_image, max_delta = 0.5, seed=None)
    new_image = tf.image.random_contrast(new_image, 0.1, 0.6, seed=None)
    # new_image = tf.image.resize_images(new_image,new_size)

    return tf.clip_by_value(new_image, -1.0, 1.0)

def flip_left_process(input):
    random_image = tf.image.flip_left_right(input)
    return random_image

def flip_up_process(input):
    random_image = tf.image.flip_up_down(input)
    return random_image

def brightness_up_process(input,alpha = 0.5):
    random_image = tf.image.adjust_brightness(input, alpha)
    return random_image

def contrast_down_process(input,alpha = -5):
    random_image = tf.image.adjust_contrast(input, alpha)
    return random_image
    

def flip_transpose(input):
    random_image = tf.image.transpose_image(input)
    return random_image

def image_resize(input,new_size):
    new_size = tf.cast(tf.convert_to_tensor(new_size),tf.int32)
    random_image = tf.image.resize_images(input,new_size)
    return random_image

def np_random_process(input):
    input_shape = input.shape
    input = np.squeeze(input)
    ## flip -1,对角， 0 垂直， 1 水平
    # random_flip = np.random.randint(-1,2)
    random_flip = 0
    image = cv2.flip(input,random_flip,dst=None)
    # ## random crop 
    # crop_size = np.random.randint(3,5) ## 2, 3
    # window_size = (crop_size * height / 4 , crop_size * weight / 4)
    # start_block = ( np.random.randint(0,height - window_size[0]), np.random.randint(0,weight - window_size[1]) )
    # image = image[start_block[0]:(start_block[0] + math.floor(window_size[0])), start_block[1]:(start_block[1] + math.floor(window_size[1]))]
     
    # resize 
    # random_height = np.random.randint(200,400)
    # random_weight = np.random.randint(200,400)
    # image = cv2.resize(image,(random_height,random_weight))

    ## change brightness
    # alpha = np.random.uniform(0.2,1.4)
    # beta = np.random.randint(0,101)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    ## write
    cv2.imwrite('test.png',image)
    return image.reshape(input_shape)
if __name__ == '__main__':
    img = cv2.imread('data/test_local/0c7ac4a8c9dfa802.png')
    np_random_process(img)
        