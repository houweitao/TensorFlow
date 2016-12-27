# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: File_util
# _date_ = 16/12/26 下午7:22

import shutil

#复制文件
# shutil.copyfile('listfile.py', 'd:/test.py')
#复制目录
# shutil.copytree('save_path/10.bak', 'save_path_bak/10.bak')
#其余可以参考shutil下的函数

def copy(_from,_to):
    shutil.copytree(_from, _to)

# copy('save_path/10.bak', 'save_path_bak/10.bak')
