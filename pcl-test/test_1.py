import pcl
import numpy as np
import pcl.pcl_visualization

def t1():
    p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    indices, model = seg.segment()
    print(p)

def t2():
    p = pcl.load("C/table_scene_lms400.pcd")
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    fil.filter().to_file("inliers.pcd")

def t3():
    p = pcl.PointCloud(10)  # "empty" point cloud
    a = np.asarray(p)       # NumPy view on the cloud
    a[:] = 0                # fill with zeros
    print(p[3])             # prints (0.0, 0.0, 0.0)
    a[:, 0] = 1             # set x coordinates to 1
    print(p[3])             # prints (1.0, 0.0, 0.0)

def t4():
    #file_path = r'E:\z\data\concategate_map.pcd'
    file_path = r'E:\Src\python-pcl\tests\tutorials\bunny.pcd'
    p = pcl.load(file_path)

    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowMonochromeCloud(p)
    v = True
    while v:
        v = not(visual.WasStopped())

if __name__ == "__main__":
    #pass
    t4()