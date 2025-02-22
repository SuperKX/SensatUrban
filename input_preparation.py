from os.path import join, exists, dirname
from sklearn.neighbors import KDTree
from tool import DataProcessing as DP
from helper_ply import write_ply
import numpy as np
import os, pickle, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='the number of GPUs to use [default: 0]')
    FLAGS = parser.parse_args()
    dataset_name = 'SensatUrban'
    dataset_path = FLAGS.dataset_path
    preparation_types = ['grid']  # Grid sampling & Random sampling
    grid_size = 0.2
    random_sample_ratio = 10
    train_files = np.sort([join(dataset_path, 'train', i) for i in os.listdir(join(dataset_path, 'train'))])  # train path
    test_files = np.sort([join(dataset_path, 'test', i) for i in os.listdir(join(dataset_path, 'test'))])  # test path
    files = np.sort(np.hstack((train_files, test_files)))  # 所有文件地址矩陣

    for sample_type in preparation_types:
        for pc_path in files:  # 逐個文件
            cloud_name = pc_path.split('/')[-1][:-4]
            print('start to process:', cloud_name)

            # create output directory
            out_folder = join(dirname(dataset_path), sample_type + '_{:.3f}'.format(grid_size))  # 文件夹名称"grid_0.200"
            os.makedirs(out_folder) if not exists(out_folder) else None

            # check if it has already calculated
            if exists(join(out_folder, cloud_name + '_KDTree.pkl')):
                print(cloud_name, 'already exists, skipped')
                continue

            if pc_path in train_files:
                xyz, rgb, labels = DP.read_ply_data(pc_path, with_rgb=True)  # 获取坐标\颜色\标签信息
            else:
                xyz, rgb = DP.read_ply_data(pc_path, with_rgb=True, with_label=False)
                labels = np.zeros(len(xyz), dtype=np.uint8)

            sub_ply_file = join(out_folder, cloud_name + '.ply')  # 输出文件名"xxx.ply"
            if sample_type == 'grid':
                sub_xyz, sub_rgb, sub_labels = DP.grid_sub_sampling(xyz, rgb, labels, grid_size)  # 下采样处理
            else:
                sub_xyz, sub_rgb, sub_labels = DP.random_sub_sampling(xyz, rgb, labels, random_sample_ratio)

            sub_rgb = sub_rgb / 255.0  # 归一化
            sub_labels = np.squeeze(sub_labels)  # 压缩维度成一维
            write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])  # 写出为ply(此处为降采样后的结果)

            search_tree = KDTree(sub_xyz, leaf_size=50)   # kdtree说明:https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
            # search_tree.get_arrays()  # 验证tree的数据xuek
            kd_tree_file = join(out_folder, cloud_name + '_KDTree.pkl')  # 输出文件名"_KDTree.pkl"
            with open(kd_tree_file, 'wb') as f:
                pickle.dump(search_tree, f)  # kdtree编码成文件

            proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
            proj_idx = proj_idx.astype(np.int32)  # 索引列表
            proj_save = join(out_folder, cloud_name + '_proj.pkl')  # 输出文件名"_proj.pkl"
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)
