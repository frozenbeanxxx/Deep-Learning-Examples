import functools
import os
import cv2
import fire
import h5py
import json
from natsort import natsorted
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from nets.inception_v1 import inception_v1, inception_v1_base
from nets.mobilenet.mobilenet_v3 import mobilenet, V3_SMALL, V3_SMALL_0_75

def t1():
    #weight_path = '/media/wx/diskE/weights/tf/inception_v1.ckpt'
    weight_path = 'E:/weights/tf/inception_v1.ckpt'
    imgH, imgW = 224, 224
    x = tf.placeholder(tf.float32, [1, imgH, imgW, 3])
    with slim.arg_scope([slim.conv2d], biases_initializer=tf.constant_initializer(0)):
        #features = inception.inception_v1(x, is_training=False)
        features = inception_v1(x, num_classes=1001, is_training=False)
        features = features[1]['AvgPool_0a_7x7']
    print(features)
    graph_def = tf.get_default_graph()
    print(graph_def)
    var_list = []
    for variable_name in tf.global_variables():
        #print(variable_name.name)
        name = variable_name.name
        if 'biases' in name:
            print(name)
        else:
            var_list.append(variable_name)

    print(var_list)
    saver = tf.train.Saver(var_list=var_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, weight_path)
        #img = cv2.imread('/media/wx/diskE/temp/c1/201911251126.jpg')
        img = cv2.imread('E:/dataset/coco/images/test2014/COCO_test2014_000000000016.jpg')
        img = img.astype(np.float32)
        img = cv2.resize(img, (imgH, imgW))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        y = sess.run(features, feed_dict={x:img})
        y = np.squeeze(y)
        print(y)
        print(y.shape)

def t2():
    weight_path = '/media/wx/diskE/weights/tf/inception_v1.ckpt'
    imgH, imgW = 224, 224
    x = tf.placeholder(tf.float32, [1, imgH, imgW, 3])
    with slim.arg_scope([slim.conv2d], biases_initializer=tf.constant_initializer(0)):
        #features = inception.inception_v1(x, is_training=False)
        features = inception_v1(x, num_classes=1001, is_training=False)
        features = features[1]['AvgPool_0a_7x7']
    print(features)
    graph_def = tf.get_default_graph()
    print(graph_def)
    var_list = []
    for variable_name in tf.global_variables():
        #print(variable_name.name)
        name = variable_name.name
        if 'biases' in name:
            print(name)
        else:
            var_list.append(variable_name)

    print(var_list)
    saver = tf.train.Saver(var_list=var_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, weight_path)

        video_path = '/media/wx/diskE/dataset/hv_browstar/videos/Screenrecorder-2019-11-20-15-02-44-979.mp4'
        if not os.path.exists(video_path):
            print(video_path, 'does not exists')
            return
        save_dir = '/media/wx/diskE/Src/video/VASNet/datasets'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # ========================= start create dataset ==================
        file_name = 'browstar_pool5.h5'
        file_path = os.path.join(save_dir, file_name)
        f = h5py.File(file_path, 'w')
        video_names = ['video_1', 'video_2']
        n_frames = 3600
        print(n_frames)
        subsample_interval = 15
        n_steps = n_frames // subsample_interval
        print(n_steps)
        picks = np.arange(0, n_frames, subsample_interval) + 2
        segments_len = 40
        num_segments = n_frames // segments_len
        n_frame_per_seg = np.full(num_segments, segments_len)
        a1 = np.arange(0, num_segments * segments_len, segments_len).reshape((-1,1))
        a2 = a1 + segments_len - 1
        change_points = np.concatenate((a1, a2), axis=1)
        num_users = 2
        user_summary = np.random.uniform(size=(num_users, n_frames))
        feature_dimension = 1024
        video_features = np.random.uniform(size=(len(video_names), n_steps, feature_dimension))
        print('video_features shape', video_features.shape)
        # ========================= end create dataset ==================

        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        loop_flag = 0
        loop_flag = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        pos = loop_flag

        show_flag = False
        if show_flag:
            cv2.namedWindow("frame", 0)
            cv2.namedWindow("bar", 0)
            cv2.createTrackbar('video_bar', 'bar', 0, frames, lambda emp: emp )

        gen_data_flag = 0
        count = 0
        while(True): 
            if show_flag:
                if loop_flag == pos:
                    loop_flag = loop_flag + 1
                    ret, frame = cap.read()
                    cv2.setTrackbarPos('video_bar', 'bar', loop_flag)
                else:
                    pos = cv2.getTrackbarPos('video_bar', 'bar')
                    loop_flag = pos
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    continue
            else:
                ret, frame = cap.read()

            img = frame.astype(np.float32)
            img = cv2.resize(img, (imgH, imgW))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            y = sess.run(features, feed_dict={x:img})
            y = np.squeeze(y)
            print(y, y.shape)
            print(count)

            name = video_names[count//n_frames]
            index = count // n_frames
            if (count-index*n_frames) % subsample_interval == 0:
                video_features[index, (count-index*n_frames) // subsample_interval] = y

            count += 1
            if count >= n_frames*len(video_names):
                break

            if(frame is None):
                break

            if show_flag:
                cv2.imshow("frame", frame)
            c = cv2.waitKey(10)

            if c & 0xFF == ord('q'):
                break

        for i, name in enumerate(video_names):
            f.create_dataset(name + '/n_frames', data=n_frames)
            f.create_dataset(name + '/n_steps', data=n_steps)
            f.create_dataset(name + '/picks', data=picks)
            f.create_dataset(name + '/n_frame_per_seg', data=n_frame_per_seg)
            f.create_dataset(name + '/change_points', data=change_points)
            f.create_dataset(name + '/user_summary', data=user_summary)
            f.create_dataset(name + '/features', data=video_features[i])
        f.close()
            
def t3():
    #weight_path = 'E:/weights/tf/v3-small_224_1.0_float/ema/'
    imgH, imgW = 224, 224
    x = tf.placeholder(tf.float32, [1, imgH, imgW, 3])
    #with slim.arg_scope():
    #    y = mobilenet(x, num_classes=1001, conv_defs=V3_SMALL)
    sess = tf.Session()
    output = mobilenet(x, num_classes=1001, conv_defs=V3_SMALL)
    output = output[1]['Predictions']
    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph('E:/weights/tf/v3-small_224_1.0_float/ema/model-388500.meta') # 加载模型结构
    saver.restore(sess, 'E:/weights/tf/v3-small_224_1.0_float/ema/model-388500')
    #saver = tf.train.import_meta_graph('E:/weights/tf/v3-small_224_1.0_float/pristine/model.ckpt-388500.meta') # 加载模型结构
    #saver.restore(sess, 'E:/weights/tf/v3-small_224_1.0_float/pristine/model.ckpt-388500')

    # names = []
    # for variable_name in tf.global_variables():
    #     name = variable_name.name
    #     #print(name)
    #     names.append(name)
    # names.sort()
    # for name in names:
    #     print(name)

    # img = cv2.imread('E:/dataset/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00000095.JPEG')
    # img = img.astype(np.float32)
    # img = cv2.resize(img, (imgH, imgW))
    # img = img / 255.0
    # img = np.expand_dims(img, axis=0)
    # y = sess.run(output, feed_dict={x:img})
    # y = np.squeeze(y)
    # print(np.argmax(y), y[np.argmax(y)])
    # print(y.shape)
    
    class_index_file = r"E:\dataset\imagenet\imagenet_class_index.json"
    with open(class_index_file, 'r') as f:
        class_dict = json.load(f)
        #print(class_dict)

    class_index_file2 = r"E:\dataset\imagenet\class_index_2.txt"
    with open(class_index_file2, 'r') as f:
        class_names = [s.strip('\n').split(' ')[0] for s in f.readlines()]
        #print(class_number)

    ground_truth_file = r"E:\dataset\imagenet\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt"
    with open(ground_truth_file, 'r') as f:
        ground_truth = [s.strip('\n') for s in f.readlines()]
        #print(ground_truth)
        print(len(ground_truth))
        count = 0
        for i, label in enumerate(ground_truth):
            image_path = 'E:/dataset/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG' % (i+1)

            #print(image_path)
            img = cv2.imread(image_path)
            img = img[:,:,::-1]
            img = img.astype(np.float32)
            img = cv2.resize(img, (int(imgH*2), int(imgW*2)))
            img = cv2.resize(img, (imgH, imgW))
            img = img / 128.0 - 1
            img = np.expand_dims(img, axis=0)
            y = sess.run(output, feed_dict={x:img})
            y = np.squeeze(y)
            y = np.argmax(y)
            #print(i+1, y-1, class_dict[str(y-1)][0], int(label), class_names[int(label)-1])

            if class_dict[str(y-1)][0] == class_names[int(label)-1]:
                count += 1
            if i % 1000 == 0:
                print(i, count/(i+1))
        print(count)

def t4():
    imgH, imgW = 224, 224
    x = tf.placeholder(tf.float32, [1, imgH, imgW, 3])
    sess = tf.Session()
    output = mobilenet(x, num_classes=1001, conv_defs=V3_SMALL_0_75, depth_multiplier=0.75)
    output = output[1]['Predictions']
    saver = tf.train.Saver()
    for variable_name in tf.global_variables():
        name = variable_name.name
        print(variable_name)
    #saver.restore(sess, 'E:/weights/tf/v3-small_224_1.0_float/ema/model-388500')
    #f = h5py.File('v3-small_224_1.0_float.h5', 'w')
    # saver.restore(sess, 'E:/weights/tf/v3-small_224_0.75_float/ema/model-497500')
    # f = h5py.File('v3-small_224_0.75_float.h5', 'w')
    # for variable_name in tf.global_variables():
    #     name = variable_name.name
    #     #print(variable_name)
    #     print(name)
    #     w = tf.get_default_graph().get_tensor_by_name(name)
    #     ww = sess.run(w)
    #     #print(w)
    #     f.create_dataset(name, data=ww)
    #     #break
    # f.close()


if __name__ == "__main__":
    fire.Fire()




