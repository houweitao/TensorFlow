# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: Help
# _date_ = 16/12/26 下午5:51

from tensorFlowStudy.InTheLab.wrong import LSTM_restore_re as restore


def work(path):
    restore.restore(path)

# work("save_path/10.bak/LSTM.ckpt")