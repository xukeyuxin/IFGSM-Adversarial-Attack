import tensorflow as tf

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
       