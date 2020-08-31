#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2020, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 28/08/20 2:37 PM
@describe:  
@version 1.0
"""


import pandas as pd
import json


class Transform:

    def __init__(self):
        self.data_origin = None  # Original data
        self.data_processed = []  # Processed data
        self.aspect_category = []  # Default aspect category
        self.polarity_symbol = {'+': 'positive', '-': 'negative', '0': 'neutral', 'c': 'conflict'}

    def get_data(self, data_path):
        """
        Retrieve original annotated data in xlsx format
        :param data_path:
        """
        self.data_origin = pd.read_excel(data_path, header=None)

    def get_text(self, row_idx):
        """
        Get comment text
        :param row_idx:
        """
        return self.data_origin.loc[row_idx, 2]

    def get_id(self, row_idx):
        """
        Get comment id
        :param row_idx:
        """
        comment_id = self.data_origin.loc[row_idx, 0]
        sentence_id = self.data_origin.loc[row_idx, 1]
        return '{}{}'.format(comment_id, sentence_id)

    def get_aspect_category(self):
        pass

    def get_aspect_term(self, row_idx):
        col_num = len(self.data_origin.columns)
        aspect_term_list = []
        for col_idx in range(3, col_num):
            aspect_term = self.data_origin.loc[row_idx, col_idx]
            if not pd.isna(aspect_term):
                try:
                    aspect, polarity = self.clean_str(aspect_term)
                    from_idx, to_idx = self.from_to_index(aspect, row_idx)
                    content = {}
                    content['polarity'] = polarity
                    content['to'] = to_idx
                    content['term'] = aspect
                    content['from'] = from_idx
                    aspect_term_list.append(content)
                except:
                    continue
        return aspect_term_list

    def clean_str(self, aspect_term):
        """

        :param aspect_term:
        :return:
        """
        tmp = aspect_term.split(']')
        aspect = tmp[0].strip('[')
        polarity = self.polarity_symbol[tmp[1].strip('[')]
        return aspect, polarity

    @staticmethod
    def count_word(word_split):
        word_num = 0
        for word in word_split:
            word_num += len(word)
        word_num += len(word_split) - 1  # Number of space
        return word_num

    @staticmethod
    def count_from_idx(aspect_idx, text_split):
        from_idx = 0
        for i in range(aspect_idx):
            from_idx += len(text_split[i])
        from_idx += aspect_idx  # Number of space
        return from_idx

    def from_to_index(self, aspect, row_idx):
        """

        :param aspect:
        :param row_idx:
        :return:
        """
        text = self.get_text(row_idx)
        text_split = text.split()  # Split text
        aspect_split = aspect.split()  # Split aspect
        try:
            aspect_idx = text_split.index(aspect_split[0])
        except:
            aspect_idx = text_split.index(aspect_split[0] + ',')
        from_idx = self.count_from_idx(aspect_idx, text_split)
        word_num = self.count_word(aspect_split)
        to_idx = from_idx + word_num
        return from_idx, to_idx

    def write(self):
        """

        """
        for row_idx in range(len(self.data_origin)):
            content = {}
            text = self.get_text(row_idx)
            text_id = self.get_id(row_idx)
            aspect_term_list = self.get_aspect_term(row_idx)
            content['text'] = text
            content['id'] = text_id
            content['opinions'] = {}
            content['opinions']['aspect_category'] = self.aspect_category
            content['opinions']['aspect_term'] = aspect_term_list
            self.data_processed.append(content)

    def write_json(self, data_path):
        """

        :param data_path:
        """
        data_processed = json.dumps(self.data_processed, indent=2)
        with open(data_path, 'w') as f:
            f.write(data_processed)
