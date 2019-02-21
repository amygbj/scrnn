# encoding: utf-8
'''
代码说明:
该方法采用了scrnn+tfrecord进行读写数据
@author: victoria
@file: srcnn_demo.py
@time: 2018/4/25 15:04
@desc:
'''
# encoding: utf-8


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import os

print(tf.__version__)

"""
SRCNN 进行图片的超频率
源输入：经过双插值的图片（低像素）
输出：图片
"""

#超参数
image_size = 96
min_after_dequeue = 10
batch_size = 5
capacity = min_after_dequeue + 3 * batch_size
learning_rate=0.001

def write_pictures(file_path,target):
    """
    写入图片
    :param file_path:
    :return:
    """
    TFwriter = tf.python_io.TFRecordWriter(target) #写入的tfrecord文件名
    image_path=os.path.join(file_path,'cut')
    #遍历目标文件夹
    for parent, dirnames, filenames in os.walk(image_path):
        for file_name in filenames:
            print(file_name)
            orgin_img ="%s/cut/%s"%(file_path,file_name)  #./data/cut/xxx.jpg
            lower_img = "%s/interpolation/%s"%(file_path,file_name)  # ./data/interpolation/xxx.jpg
            target = Image.open(orgin_img).tobytes()
            input = Image.open(lower_img).tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label":tf.train.Feature(bytes_list=tf.train.BytesList(value=[target])),  #超分辨率的target为它同size的清晰图片
                "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[input]))  #超分辨率的input为它同size的模糊图片
            }) )
            TFwriter.write(example.SerializeToString())

    TFwriter.close()

def get_image(image_path, mode='RGB'):
    """
    Read image from image_path
    :param image_path: Path of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)
    return np.array(image.convert(mode))


def buid_input(tfrecords_filename):
    """
    该方法用于提供inputs和target
    :param tfrecords_filename:
    :return:
    """
    # 写入数据

    write_pictures('./data/faces', tfrecords_filename)
    print('write success')

    fileNameQue = tf.train.string_input_producer([tfrecords_filename])

    # 读取文件
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fileNameQue)  # 返回文件名和文件

    # 解析读取的样例。
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img': tf.FixedLenFeature([], tf.string)
                                       })  # 取出包含image和label的feature对象

    decoded_images = tf.decode_raw(features['img'],tf.uint8)
    decoded_images.set_shape(image_size*image_size*3)
    decoded_images = tf.cast(decoded_images,tf.float32)
    labels = tf.decode_raw(features['label'],tf.uint8)  # target (96,96,3)
    labels.set_shape(image_size*image_size*3)
    labels = tf.cast(labels,tf.float32)
    print('decoded_images.shape %s' %decoded_images.shape)
    print(labels.shape)
    images_input = tf.reshape(decoded_images, [image_size ,image_size,3])  # 图片shape
    images_target = tf.reshape(labels, [image_size ,image_size,3])

    image_batch, label_batch = tf.train.shuffle_batch([images_input, images_target],
                                                      batch_size=batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch


def build_simple_net(input):
    """
    构建单层神经网络
    :param input:  单个通道的输入
    :param target:  单个通道的输出
    :return:
    """

    #layer1
    layer1 = tf.layers.conv2d(inputs=input,filters=64,kernel_size=9,strides=1,padding='SAME')
    layer1 = tf.nn.relu(layer1)

    #layer2
    layer2 = tf.layers.conv2d(inputs=layer1,filters=32,kernel_size=1,strides=1,padding='SAME')
    layer2 = tf.nn.relu(layer2)

    #layer3
    layer3 = tf.layers.conv2d(inputs=layer2,filters=1,kernel_size=5,strides=1,padding='SAME')
    layer3 = tf.nn.relu(layer3)

    return layer3

def build_network(tfrecords_filename):
    """
    构造整个神经网络
    :param input_r:
    :param input_g:
    :param input_b:
    :return:
    """

    inputs, target = buid_input(tfrecords_filename) #构造输入
    print(inputs.get_shape()) #(b,96,96,3)

    #分割成单通道
    input_b,input_g,input_r = tf.split(inputs, 3, axis=3)
    print(input_b.get_shape())

    #单通道网络输出
    logit_r = build_simple_net(input_r)
    logit_g = build_simple_net(input_g)
    logit_b = build_simple_net(input_b)

    #合并输出
    logits = tf.concat((logit_b,logit_g,logit_r),axis=3)

    #buid loss
    loss = tf.reduce_mean(tf.square(logits - target))

    #buid optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    return loss,optimizer,accuracy



tfrecords_filename = "./data/aigirls.tfrecords"
loss, optimizer ,accuracy  = build_network(tfrecords_filename)
init = tf.initialize_all_variables()

epochs = 10

with tf.Session() as sess :
    sess.run(init)
    print('train')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(epochs) :
        print("epochs %s , loss %s " % (i, sess.run(loss)))
        print("epochs %s , accuracy %s " % (i, sess.run(accuracy)))
        sess.run(optimizer)
    coord.request_stop()
    coord.join(threads)

