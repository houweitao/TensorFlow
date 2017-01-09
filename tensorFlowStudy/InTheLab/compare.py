# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: compare
# _date_ = 16/12/13 上午1:29

from __future__ import division
import string


def compare_five(event_prediction, event_real, type_prediction, type_real, polarity_prediction, polarity_real,
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
        # print(batch_index)
        i = 0
        step_size = len(event_prediction[batch_index])
        while i < step_size:
            if string.atof(event_real[batch_index][i][2]) == 1:
                continue
            if string.atof(event_real[batch_index][i][0]) == 1:
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

    return same_event / all, same_type / all, same_polarity / all, same_degree / all, same_modality / all


def compare_five_p_and_r(event_prediction, event_real, type_prediction, type_real, polarity_prediction, polarity_real,
                         degree_prediction, degree_real, modality_prediction, modality_real):
    # batch size
    batch_size = len(event_prediction)

    all = 0
    same_event = 0
    same_type = 0
    same_polarity = 0
    same_degree = 0
    same_modality = 0

    prediction_event = 0
    prediction_event_type = 0
    prediction_event_polarity = 0
    prediction_event_degree = 0
    prediction_event_modality = 0

    real_event = 0
    real_type = 0
    real_polarity = 0
    real_degree = 0
    real_modality = 0

    # wrong_event = 0
    #
    tp_event = 0
    fp_event = 0
    fn_event = 0
    tn_event = 0
    #
    # tp_type = 0
    # fp_type = 0
    # fn_type = 0
    # tn_type = 0
    #
    # tp_polarity = 0
    # fp_polarity = 0
    # fn_polarity = 0
    # tn_polarity = 0
    #
    # tp_degree = 0
    # fp_degree = 0
    # fn_degree = 0
    # tn_degree = 0
    #
    # tp_modality = 0
    # fp_modality = 0
    # fn_modality = 0
    # tn_modality = 0

    batch_index = 0
    while batch_index < batch_size:
        i = 0
        step_size = len(event_prediction[batch_index])
        while i < step_size:
            event_p = event_prediction[batch_index][i]
            event_r = event_real[batch_index][i]

            type_p = type_prediction[batch_index][i]
            type_r = type_real[batch_index][i]

            polarity_p = polarity_prediction[batch_index][i]
            polarity_r = polarity_real[batch_index][i]

            degree_p = degree_prediction[batch_index][i]
            degree_r = degree_real[batch_index][i]

            modality_p = modality_prediction[batch_index][i]
            modality_r = modality_real[batch_index][i]

            # is no word
            if string.atof(event_r[0]) == 1:
                break

            # is event
            pre_pos = get_max_expect_first(event_p)
            real_pos = get_max_expect_first(event_r)

            if pre_pos == 1:
                prediction_event += 1
            if real_pos == 1:
                real_event += 1
            if pre_pos == 1 & real_pos == 1:
                same_event += 1

                if get_max(type_p) == get_max(type_r):
                    same_type += 1
                if get_max(polarity_p) == get_max(polarity_r):
                    same_polarity += 1
                if get_max(degree_p) == get_max(degree_r):
                    same_degree += 1
                if get_max(modality_p) == get_max(modality_r):
                    same_modality += 1

            i += 1
        batch_index += 1

    print ('event', same_event, prediction_event, real_event)
    print ('type', same_type, prediction_event, real_event)
    print ('polarity', same_polarity, prediction_event, real_event)
    print ('degree', same_degree, prediction_event, real_event)
    print ('modality', same_modality, prediction_event, real_event)

    precision_event = same_event / prediction_event
    recall_event = same_event / real_event
    combine_event = '(' + str(precision_event) + ',' + str(recall_event) + ')'

    precision_type = same_type / prediction_event
    recall_type = same_type / real_event
    combine_type = '(' + str(precision_type) + ',' + str(recall_type) + ')'

    precision_polarity = same_polarity / prediction_event
    recall_polarity = same_polarity / real_event
    combine_polarity = '(' + str(precision_polarity) + ',' + str(recall_polarity) + ')'

    precision_degree = same_degree / prediction_event
    recall_degree = same_degree / real_event
    combine_degree = '(' + str(precision_degree) + ',' + str(recall_degree) + ')'

    precision_modality = same_modality / prediction_event
    recall_modality = same_modality / real_event
    combine_modality = '(' + str(precision_modality) + ',' + str(recall_modality) + ')'

    return combine_event, combine_type, combine_polarity, combine_degree, combine_modality  # def compare_event(prediction, real):


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


def get_max_expect_first(nums):
    max = -2000000
    pos = -1

    i = 1
    while i < len(nums):
        if (nums[i] > max):
            pos = i
            max = nums[i]

        i += 1

    return pos

# nums = [-0.2, 0, 0.1, -0.4]
# print(get_max(nums))
