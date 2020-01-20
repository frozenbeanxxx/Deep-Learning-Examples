import h5py
from natsort import natsorted
import numpy as np 



def t1():
    file_name = 'test.h5'
    f = h5py.File(file_name, 'w')
    f.create_group('/grp1')
    f.close()

    f = h5py.File(file_name, 'r')
    keys = f.keys()
    for key in keys:
        print(key)
    print(f.keys())
    #print(keys[0])
    print(f['/grp1'])

def t2():
    file_name = 'test.h5'
    arr = np.arange(12, dtype=np.float32)
    arr = arr.reshape((3,4))
    f = h5py.File(file_name, 'w')
    f.create_dataset('t2', data=arr)
    f.create_dataset('t3', data=file_name)
    f.close()

    f = h5py.File(file_name, 'r')
    keys = f.keys()
    for key in keys:
        print(key)
    print(f.keys())
    #print(keys[0])
    print(f['t2'])
    print(f['t2'][:])
    print(f['t2'][()])
    print(f['t3'])
    print(f['t3'][()])

def t3():
    #h5_file_name = 'eccv16_dataset_summe_google_pool5.h5'
    #h5_file_name = 'browstar_summe_pool5.h5'
    h5_file_name = 'browstar_pool5.h5'
    f = h5py.File(h5_file_name, 'r')
    print(f)
    keys = f.keys()
    keys = natsorted(keys)
    print(keys)
    for key in keys:
        #print(key)
        a = f[key]
        print(a.keys())
        for key2 in a.keys():
            print(a[key2])
            print(a[key2][...])
        #print(a['change_points'][:])
        #print(a['gtscore'][:])
            
    f.close()

def t4():
    file_name = 'browstar_pool5.h5'
    f = h5py.File(file_name, 'w')
    video_names = ['video_1', 'video_2']
    for name in video_names:
        n_frames = 3600
        print(n_frames)
        f.create_dataset(name + '/n_frames', data=n_frames)

        subsample_interval = 15
        n_steps = n_frames // subsample_interval
        print(n_steps)
        f.create_dataset(name + '/n_steps', data=n_steps)

        picks = np.arange(0, n_frames, subsample_interval) + 2
        f.create_dataset(name + '/picks', data=picks)

        segments_len = 40
        num_segments = n_frames // segments_len
        n_frame_per_seg = np.full(num_segments, segments_len)
        f.create_dataset(name + '/n_frame_per_seg', data=n_frame_per_seg)

        a1 = np.arange(0, num_segments * segments_len, segments_len).reshape((-1,1))
        a2 = a1 + segments_len - 1
        change_points = np.concatenate((a1, a2), axis=1)
        f.create_dataset(name + '/change_points', data=change_points)

        num_users = 2
        user_summary = np.random.uniform(size=(num_users, n_frames))
        f.create_dataset(name + '/user_summary', data=user_summary)

        feature_dimension = 1024
        features = np.random.uniform(size=(n_steps, feature_dimension))
        f.create_dataset(name + '/features', data=features)

        #gtscore = np.random.uniform(size=(n_steps))
        #f.create_dataset(name + '/gtscore', data=gtscore)

        #f.create_dataset(name + '/gtsummary', data=data_of_name)
        #f.create_dataset(name + '/video_name', data=data_of_name)
    f.close()

    f = h5py.File(file_name, 'r')
    print(f)
    keys = f.keys()
    keys = natsorted(keys)
    print(keys)
    for key in keys:
        #print(key)
        a = f[key]
        print(a.keys())
        for key2 in a.keys():
            b = a[key2][...]
            print(key2, b.shape)

def t4():
    #file_name = '/media/wx/diskE/weights/keras/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    file_name = '/media/wx/diskE/weights/keras/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    f = h5py.File(file_name, 'r')
    def print_name(name):
        print(name)
    f.visit(print_name)
    #print(f)
    keys = f.keys()
    keys = natsorted(keys)
    #print(keys)
    count = 1
    for i, key in enumerate(keys[:]):
        a = f[key]
        print('\n', key, a)
        b = a.keys()
        if len(b) != 0:
            print(count)
            count += 1
        for bb in b:
            print(a[bb])


if __name__ == "__main__":
    t4()
