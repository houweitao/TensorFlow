# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: count_p_r_of_CRF
# _date_ = 17/1/10 上午12:09

import re

root = "/Users/hou/Dropbox/Graduate/SemEval_2017_task12/open_source/CRF++-0.58/example/mine/"

event_file = "result_event"
type_file = "result_type"
polarity_file = "result_polarity"
degree_file = "result_degree"
modality_file = "result_modality"


def read(filename, root=root):
    f = open(root + filename, "r")
    lines = f.readlines()  # 读取全部内容

    per = 0.0
    real = 0.0
    right = 0.0

    for line in lines:
        line = line[:-1]
        line.replace("NO_EVENT", "NO", 1)
        # print(line)

        list = re.split('\t', line)
        # print(list)

        if len(list) == 3:
            if list[1] != "NO":
                per += 1
            if list[2] != "NO":
                real += 1
            if list[1] == list[2]:
                if list[2] != "NO":
                    right += 1

    print (per, real, right)
    return right / per, right / real


p, r = read(event_file)
print(p, r)
p, r = read(type_file)
print(p, r)
p, r = read(polarity_file)
print(p, r)
p, r = read(degree_file)
print(p, r)
p, r = read(modality_file)
print(p, r)
