# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: load_data_embding
# _date_ = 16/12/11 下午7:31

train = "/Users/hou/Documents/data/embding/train/"
test = "/Users/hou/Documents/data/embding/test/"
dev = "/Users/hou/Documents/data/embding/dev/"
# root = "/Users/hou/Documents/data/embding/train2/"
# root = "/Users/hou/Documents/data/embding/train5000/"

import os
import random
import numpy as np


def all_train_filename():
    return os.listdir(r'/Users/hou/Documents/data/embding/train/')


def all_test_filename():
    return os.listdir(r'/Users/hou/Documents/data/embding/test/')


def all_dev_filename():
    return os.listdir(r'/Users/hou/Documents/data/embding/dev/')


def read(filename, root):
    f = open(root + filename, "r")
    lines = f.readlines()  # 读取全部内容

    word = []
    event = []
    type = []
    polarity = []
    degree = []
    modality = []

    count = 0
    for line in lines:
        line = line[:-1]

        if (count % 6 == 0):
            word.append(line.split(','))

        if (count % 6 == 1):
            event.append(line.split(','))

        if (count % 6 == 2):
            type.append(line.split(','))

        if (count % 6 == 3):
            polarity.append(line.split(','))

        if (count % 6 == 4):
            degree.append(line.split(','))

        if (count % 6 == 5):
            modality.append(line.split(','))

        # print (line)
        count = count + 1
        # if (count == 10):
        #     break

    word = normalize_form(word)
    event = normalize_form(event)
    type = normalize_form(type)
    polarity = normalize_form(polarity)
    degree = normalize_form(degree)
    modality = normalize_form(modality)

    return word, event, type, polarity, degree, modality


def normalize_form(list):
    list = np.array(list)
    # list = list.astype(float)
    return list


def get_random_train_filename(n):
    file_list = all_train_filename()

    list = []
    for file in file_list:
        list.append(file)

    random.shuffle(list)
    ret = []
    pos = 0
    while len(ret) < n:
        ret.append(list[pos])
        pos += 1
    # ret = normalize_form(ret)
    print(ret)
    return ret


def get_random_dev_filename(n):
    file_list = all_dev_filename()

    list = []
    for file in file_list:
        list.append(file)

    random.shuffle(list)
    ret = []
    pos = 0
    while len(ret) < n:
        ret.append(list[pos])
        pos += 1
    # ret = normalize_form(ret)
    print(ret)
    return ret


def get_random_test_filename(n):
    file_list = all_test_filename()

    list = []
    for file in file_list:
        list.append(file)

    random.shuffle(list)
    ret = []
    pos = 0
    while len(ret) < n:
        ret.append(list[pos])
        pos += 1
    # ret = normalize_form(ret)
    print(ret)
    return ret


def get_train_batches(n=3):
    words = []
    events = []
    types = []
    polarities = []
    degrees = []
    modalities = []
    files = get_random_train_filename(n)

    for file in files:
        word, event, type, polarity, degree, modality = read(file, train)
        words.append(word)
        events.append(event)
        types.append(type)
        polarities.append(polarity)
        degrees.append(degree)
        modalities.append(modality)

    words = normalize_form(words)
    events = normalize_form(events)
    types = normalize_form(types)
    polarities = normalize_form(polarities)
    degrees = normalize_form(degrees)
    modalities = normalize_form(modalities)

    return words, events, types, polarities, degrees, modalities


def get_test_batches(n=3):
    words = []
    events = []
    types = []
    polarities = []
    degrees = []
    modalities = []
    files = get_random_test_filename(n)

    for file in files:
        word, event, type, polarity, degree, modality = read(file, test)
        words.append(word)
        events.append(event)
        types.append(type)
        polarities.append(polarity)
        degrees.append(degree)
        modalities.append(modality)

    words = normalize_form(words)
    events = normalize_form(events)
    types = normalize_form(types)
    polarities = normalize_form(polarities)
    degrees = normalize_form(degrees)
    modalities = normalize_form(modalities)

    return words, events, types, polarities, degrees, modalities


def get_dev_batches(n=3):
    words = []
    events = []
    types = []
    polarities = []
    degrees = []
    modalities = []
    files = get_random_dev_filename(n)

    for file in files:
        word, event, type, polarity, degree, modality = read(file, dev)
        words.append(word)
        events.append(event)
        types.append(type)
        polarities.append(polarity)
        degrees.append(degree)
        modalities.append(modality)

    words = normalize_form(words)
    events = normalize_form(events)
    types = normalize_form(types)
    polarities = normalize_form(polarities)
    degrees = normalize_form(degrees)
    modalities = normalize_form(modalities)

    return words, events, types, polarities, degrees, modalities

# def work():
#     for filename in get_random_filename(10):
#         word, event, type, polarity, degree, modality = read(filename)
#         print(degree)
#         print(filename)

# for filename in all_filename():
#     word, event, type, polarity, degree, modality = read(filename)
#
#     print(event)
#     print(len(event))
#     break


# work()

# words, events, types, polarities, degrees, modalities = get_batches(3)
# print(events.shape)
# print(types.shape)
# print(polarities.shape)
# print(degrees.shape)
# print(modalities.shape)
# # print(words)
# print(words.shape)
