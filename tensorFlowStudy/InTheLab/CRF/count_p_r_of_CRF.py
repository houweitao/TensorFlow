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


def get_lines(filename, root=root):
    f = open(root + filename, "r")

    ret = []

    lines = f.readlines()  # 读取全部内容
    for line in lines:
        line = line.replace("NO_EVENT", "NO", 2)
        line = line[:-1]
        list = re.split('\t', line)
        if len(list) == 3:
            ret.append(list[2])
        else:
            ret.append('')
            # print(list[2])

    return ret


def get_p_r(real_event_list, result_filename):
    f = open(root + result_filename, "r")
    lines = f.readlines()  # 读取全部内容

    # print(len(real_event_list), len(lines))
    per = 0.0
    real = 0.0
    right = 0.0

    real_index = 0
    for line in lines:
        line = line[:-1]
        line = line.replace("NO_EVENT", "NO")
        # print(line)

        list = re.split('\t', line)
        # print(list)

        if len(list) == 3:
            if list[1] != "NO":
                real += 1
            if list[2] != "NO":
                per += 1
            if list[1] == list[2]:
                if list[2] != "NO":
                    if real_event_list[real_index] != "NO":
                        right += 1
        real_index += 1

    print (per, real, right)
    return right / per, right / real


def read(filename, root=root):
    f = open(root + filename, "r")
    lines = f.readlines()  # 读取全部内容

    per = 0.0
    real = 0.0
    right = 0.0

    for line in lines:
        line = line[:-1]
        line = line.replace("NO_EVENT", "NO", 2)
        # print(line)

        list = re.split('\t', line)
        # print(list)

        if len(list) == 3:
            if list[1] != "NO":
                real += 1
            if list[2] != "NO":
                per += 1
            if list[1] == list[2]:
                if list[2] != "NO":
                    right += 1
                    # right += 1

    print (per, real, right)
    return right / per, right / real


def num(filename):
    f = open(root + filename, "r")
    lines = f.readlines()  # 读取全部内容
    return len(lines)


# p, r = read(event_file)
# print(p, r)
# p, r = read(type_file)
# print(p, r)
# p, r = read(polarity_file)
# print(p, r)
# p, r = read(degree_file)
# print(p, r)
# p, r = read(modality_file)
# print(p, r)
#
# get_lines(event_file)

# print(num(event_file))
# print(num(type_file))
# print(num(polarity_file))
# print(num(degree_file))
# print(num(modality_file))

per_event_list = get_lines(event_file)

p, r = get_p_r(per_event_list, event_file)
print(p, r)
p, r = get_p_r(per_event_list, type_file)
print(p, r)
p, r = get_p_r(per_event_list, polarity_file)
print(p, r)
p, r = get_p_r(per_event_list, degree_file)
print(p, r)
p, r = get_p_r(per_event_list, modality_file)
print(p, r)
