# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: load_data
# _date_ = 16/12/8 下午4:57

# root = "/Users/hou/Documents/data/train/"
root = "/Users/hou/Documents/data/train2/"

import os
import random
import numpy as np


def all_filename():
    return os.listdir(r'/Users/hou/Documents/data/train2/')


def read(filename):
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
            word.append(list(line))

        if (count % 6 == 1):
            event.append(list(line))

        if (count % 6 == 2):
            type.append(list(line))

        if (count % 6 == 3):
            polarity.append(list(line))

        if (count % 6 == 4):
            degree.append(list(line))

        if (count % 6 == 5):
            modality.append(list(line))

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

    # word = np.array(word)
    # event = np.array(event)
    # type = np.array(type)
    # polarity = np.array(polarity)
    # degree = np.array(degree)
    # modality = np.array(modality)

    return word, event, type, polarity, degree, modality


def normalize_form(list):
    list = np.array(list)
    # list = list.astype(float)
    return list


def get_random_filename(n):
    file_list = all_filename()

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


def get_batches(n=3):
    words = []
    events = []
    types = []
    polarities = []
    degrees = []
    modalities = []

    files = get_random_filename(n)

    for file in files:
        word, event, type, polarity, degree, modality = read(file)
        words.append(word)
        events.append(event)
        types.append(type)
        polarities.append(polarity)
        degrees.append(degree)
        modalities.append(modality)

    words=normalize_form(words)
    events=normalize_form(events)
    types=normalize_form(types)
    polarities=normalize_form(polarities)
    degrees=normalize_form(degrees)
    modalities=normalize_form(modalities)

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

words, events, types, polarities, degrees, modalities = get_batches(1)
print(types.shape)
print(words.shape)
