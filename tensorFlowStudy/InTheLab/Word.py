# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: Word
# _date_ = 16/12/13 上午2:33

class Word:
    event = []
    type = []
    polarity = []
    degree = []
    modality = []

    def __init__(self, event, type, polarity, degree, modality):
        self.event = event
        self.type = type
        self.polarity = polarity
        self.degree = degree
        self.modality = modality
