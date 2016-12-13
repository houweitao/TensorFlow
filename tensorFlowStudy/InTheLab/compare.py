# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: compare
# _date_ = 16/12/13 上午1:29

from __future__ import division

def compare_all(event_prediction, event_real, type_prediction, type_real, polarity_prediction, polarity_real,
                degree_prediction, degree_real, modality_prediction, modality_real):
    # batch size
    batch_size = len(event_prediction)

    all = 0
    same_event = 0
    same_type = 0
    same_polarity = 0
    same_degree = 0
    same_modality = 0

    batch_index = 0
    while batch_index < batch_size:
        i = 0
        step_size = len(event_prediction[batch_index])
        while i < step_size:
            if event_real[batch_index][i][2] == 1:
                continue
            else:
                all += 1

                event_p = event_prediction[batch_index][i]
                event_r = event_real[batch_index][i]

                if get_max(event_p) == get_max(event_r):
                    same_event += 1

                type_p = type_prediction[batch_index][i]
                type_r = type_real[batch_index][i]

                if get_max(type_p) == get_max(type_r):
                    same_type += 1

                polarity_p = polarity_prediction[batch_index][i]
                polarity_r = polarity_real[batch_index][i]

                if get_max(polarity_p) == get_max(polarity_r):
                    same_polarity += 1

                degree_p = degree_prediction[batch_index][i]
                degree_r = degree_real[batch_index][i]

                if get_max(degree_p) == get_max(degree_r):
                    same_degree += 1

                modality_p = modality_prediction[batch_index][i]
                modality_r = modality_real[batch_index][i]

                if get_max(modality_p) == get_max(modality_r):
                    same_modality += 1

            i += 1
        batch_index += 1


    return same_event / all, same_type / all, same_polarity / all, same_degree / all, same_modality / all  # def compare_event(prediction, real):


#     len = len(prediction)
#
#     all = 0
#     same = 0
#
#     i = 0
#     while i < len:
#         index_need = get_max(real[i])
#         index_prediction = get_max(prediction[i])
#
#         if index_need != 0:
#             all += 1
#             if (index_need == index_prediction):
#                 same += 1
#         i += 1
#
#     return


def get_max(nums):
    max = -2000000
    pos = -1

    i = 0
    while i < len(nums):
        if (nums[i] > max):
            pos = i
            max = nums[i]

        i += 1

    return pos

# nums = [-0.2, 0, 0.1, -0.4]
# print(get_max(nums))
