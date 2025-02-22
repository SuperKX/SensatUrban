
# Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges

This is the official repository of the **SensatUrban** dataset. For technical details, please refer to:

**Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges** <br />
[Qingyong Hu](https://qingyonghu.github.io/), [Bo Yang*](https://yang7879.github.io/), [Sheikh Khalid](https://uk.linkedin.com/in/fakharkhalid), 
[Wen Xiao](https://www.ncl.ac.uk/engineering/staff/profile/wenxiao.html), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/). <br />
**[[Paper](http://arxiv.org/abs/2009.03137)] [[Blog](https://zhuanlan.zhihu.com/p/259208850)] [[Video](https://www.youtube.com/watch?v=IG0tTdqB3L8)] [[Project page](https://github.com/QingyongHu/SensatUrban)] [[Download](https://forms.gle/m4HJiqZxnq8rmjc8A)] 
[[Evaluation](https://competitions.codalab.org/competitions/31519#participate-submit_results)]** <br />

🔥 原始说明此处不再赘述。本文档仅用于记录配置使用情况。


## Environment Configuration （Ubuntu18）    

### 0、环境说明 ###
本人环境是UBUNTU18 + CUDA10.0 + cudnn7.4 + tensorflow1.14

🔥 注意：    
1、python3.6+cuda10+tensorflow1.14验证跑通CPU运行环境。    
2、python3.6+cuda11（RTX3090）+tensorflow2.6+keras2.6下配置跑通GPU运行环境，代码需要修改tensorflow库调用，改用tensorflow1版本的书写方式调用。RTX30系显卡支持最低cuda版本为cuda11，tensorflow1.14不支持cuda11。

### 1、下载数据集 ###
Download the files named "data_release.zip" [here](https://forms.gle/m4HJiqZxnq8rmjc8A). 
本人目录结构：`Randlanet（即代码根目录）/Dataset/SensatUrban/train & test`.

### 2、创建环境 ###
```
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
```
🔥 这里没找到pyyaml的5.4版本，直接执行的“pip install pyyaml”，默认安装了5.3.1版本。  

执行以下批处理，处理cpp库
```
sh compile_op.sh
```
🔥 注意：    
1、安装g++和gcc，保证两者版本一致。    
2、utils文件夹需要在正确位置上，否则生成库有问题。    
3、经过验证，python版本不同编译的库不可通用，如python3.5版本编译的不可再python3.6下使用；但ubuntu18编译的库可在ubuntu20下使用。
### 3、数据预处理 ###
执行批处理之前，需要安装以下库，否则会出错：  
```
pip install cython
pip install numpy
pip install sklearn
pip install open3d
pip install tensorflow-gpu==1.14
```
如下运行，执行数据预处理。
```
python input_preparation.py --dataset_path $YOURPATH
cd $YOURPATH
cd ../
mkdir original_block_ply
mv data_release/train/* original_block_ply
mv data_release/test/* original_block_ply
mv data_release/grid* ./
```
 
The data should organized in the following format:
```
/Dataset/SensatUrban/
          └── original_block_ply/
                  ├── birmingham_block_0.ply
                  ├── birmingham_block_1.ply 
		  ...
	    	  └── cambridge_block_34.ply 
          └── grid_0.200/
	     	  ├── birmingham_block_0_KDTree.pkl
                  ├── birmingham_block_0.ply
		  ├── birmingham_block_0_proj.pkl 
		  ...
	    	  └── cambridge_block_34.ply 
```
### 4、训练 ###
- Start training: (Please first modified the root_path)
```
python main_SensatUrban.py --mode train --gpu 0 
```
🔥 训练前，记得修改class SensatUrban的root_path参数，ubuntu下本人地址为'/home/username/桌面/Randlanet/Dataset'    
🔥 使用cpu则修改参数为“--gpu -1”    
🔥 本机上tool.py中注释了cpu和gpu不同训练超参数。    
🔥 batchsize过小，结果显著较差，miou基本维持在10%左右。

### 5、测试 ###
- Evaluation:
```
python main_SensatUrban.py --mode test --gpu 0 
```
- Submit the results to the server:
The compressed results can be found in `/test/Log_*/test_preds/submission.zip`. Then, feel free to submit this results to the 
[evaluation server](https://competitions.codalab.org/competitions/31519#participate-submit_results). 

### 6、可视化 ###
增加了可视化部分的代码，只需要命令行运行：
```
python visualize2ply.py
```
🔥 可视化处理前请自行需要根据需求修改文件地址。    
🔥 可视化结果可以通过cloudcompare读入ply文件，scalar选择class一项。

### Citation
If you find our work useful in your research, please consider citing:

	@inproceedings{hu2020towards,
	  title={Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges},
	  author={Hu, Qingyong and Yang, Bo and Khalid, Sheikh and Xiao, Wen and Trigoni, Niki and Markham, Andrew},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  year={2021}
	}


### Updates
* 01/03/2021: The SensatUrban has been accepted by CVPR 2021!
* 11/02/2021: The dataset is available for download!
* 07/09/2020: Initial release!


## Related Repos
1. [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/RandLA-Net.svg?style=flat&label=Star)
2. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)
4. [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SpinNet.svg?style=flat&label=Star)
5. [SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds with 1000x Fewer Labels](https://github.com/QingyongHu/SQN) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SQN.svg?style=flat&label=Star)



