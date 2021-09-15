import numpy as np
from tool import DataProcessing as DP
from helper_ply import write_ply, read_ply  # xuek

# 颜色标签
label_colors = [[85, 107, 47], [0, 255, 0],  # tree -> Green
                [255, 165, 0],  # building -> orange
                [41, 49, 101],  # Walls ->  darkblue
                [0, 0, 0],  # Bridge -> black
                [0, 0, 255],  # parking -> blue
                [255, 0, 255],  # rail -> Magenta
                [200, 200, 200],  # traffic Roads ->  grey
                [89, 47, 95],  # Street Furniture  ->  DimGray
                [255, 0, 0],  # cars -> red
                [255, 255, 0],  # Footpath  ->  deeppink
                [0, 255, 255],  # bikes -> cyan
                [0, 191, 255]  # water ->  skyblue
                ]
label_color = np.asarray(label_colors, dtype=np.uint8)
# 　label_color = label_color / 255.0

# 地址
# 1\读取的ply地址
inputplypath = ['Dataset/SensatUrban/original_block_ply/birmingham_block_2.ply',
                'Dataset/SensatUrban/original_block_ply/birmingham_block_8.ply',
                'Dataset/SensatUrban/original_block_ply/cambridge_block_15.ply',
                'Dataset/SensatUrban/original_block_ply/cambridge_block_16.ply',
                'Dataset/SensatUrban/original_block_ply/cambridge_block_22.ply',
                'Dataset/SensatUrban/original_block_ply/cambridge_block_27.ply']
# 2\读取的标签地址
inputlabelpath = ['test/Log_2021-09-05_11-05-28/test_preds/birmingham_block_2.label',
                  'test/Log_2021-09-05_11-05-28/test_preds/birmingham_block_8.label',
                  'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_15.label',
                  'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_16.label',
                  'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_22.label',
                  'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_27.label']
# 2\写出的ply地址
outputplypath = ['test/Log_2021-09-05_11-05-28/test_preds/birmingham_block_2.ply',
                 'test/Log_2021-09-05_11-05-28/test_preds/birmingham_block_8.ply',
                 'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_15.ply',
                 'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_16.ply',
                 'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_22.ply',
                 'test/Log_2021-09-05_11-05-28/test_preds/cambridge_block_27.ply']

# 数据处理
for i in range(0, 6):
    xyz, rgb = DP.read_ply_data(inputplypath[i], with_rgb=True,
                                with_label=False)  # xyz.astype(np.float32), rgb.astype(np.uint8)
    label = np.fromfile(inputlabelpath[i], dtype=np.uint8)
    write_ply(outputplypath[i], [xyz, rgb, label], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])  # xuek
