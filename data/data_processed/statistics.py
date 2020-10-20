#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2020, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 20/10/20 10:15 AM
@describe:  
@version 1.0
"""


import json
import matplotlib.pyplot as plt


class Sample:

    def __init__(self, train):
        self.file_name = 'TripAdvisor_hotel_{}.json'.format(train)
        with open(self.file_name) as f:
            self.data = json.load(f)
        self.num_pos, self.num_neu, self.num_neg = 0, 0, 0

    def sum_polarity(self):
        """
        Summarize polarity
        """
        for content in self.data:
            try:
                for aspect_term in content['opinions']['aspect_term']:
                    if aspect_term['polarity'] == 'positive':
                        self.num_pos += 1
                    elif aspect_term['polarity'] == 'neutral':
                        self.num_neu += 1
                    elif aspect_term['polarity'] == 'negative':
                        self.num_neg += 1
            except:
                continue
        print(self.num_pos, self.num_neu, self.num_neg)


if __name__ == '__main__':
    s = Sample('test')
    s.sum_polarity()
