from distutils.core import setup
from distutils.extension import Extension  # distutils用于写扩展模块：https://cloud.tencent.com/developer/article/1531903
from Cython.Distutils import build_ext  # cython编译：https://blog.csdn.net/jay_yxm/article/details/106679075
import numpy

ext_modules = [Extension(
    "nearest_neighbors",  # 发布的模块名
    sources=["knn.pyx", "knn_.cxx", ],  # 源文件
    include_dirs=["./", numpy.get_include()],  # 包含目录
    language="c++",  # 语言
    extra_compile_args=["-std=c++11", "-fopenmp", ],  # “c++版本”、“多线程”：额外编译信息传递给Cython编译器
    extra_link_args=["-std=c++11", '-fopenmp'],
)]

setup(
    name="KNN NanoFLANN",  # 包名称
    ext_modules=ext_modules,  # 指定拓展模块：参数用于构建 C 和 C++ 扩展扩展包
    cmdclass={'build_ext': build_ext},  # 添加自定义命令
)
# setup参数说明：https://blog.konghy.cn/2018/04/29/setup-dot-py/
# 查看命令行相应指令含义： E:\SensatUrban\utils\nearest_neighbors>python setup.py --help-commands
# 若编译成功，则在目录下会出现两个文件： xxx.c，xxx.pyd（若在Linux平台下会出现hello.so），此时.so文件或者.pyd文件就可以像普通的python文件一样，被import。
