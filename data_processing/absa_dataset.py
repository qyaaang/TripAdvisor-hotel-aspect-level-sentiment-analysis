#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2020, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 31/08/20 3:47 PM
@describe:  
@version 1.0
"""


import json


class ABSA:

    def __init__(self):
        self.dataset_list = []
        self.data = []

    def set_dataset_name(self, dataset_list):
        """

        :param dataset_list:
        """
        self.dataset_list = dataset_list

    def pack_data(self, data_path, dataset='train'):
        """

        :param data_path:
        :param dataset:
        """
        for dataset_name in self.dataset_list:
            file_name = '{}-{}.json'.format(dataset_name, dataset)
            with open(data_path + file_name, 'r') as f:
                tmp = json.load(f)
                self.data += tmp

    def write_json(self, save_path, dataset='train'):
        """

        :param save_path:
        :param dataset:
        """
        data = json.dumps(self.data, indent=2)
        file_name = 'ABSA_restaurants_{}.json'.format(dataset)
        with open(save_path + file_name, 'w') as f:
            f.write(data)


if __name__ == '__main__':
    with open('/Users/qunyang/Dropbox (Personal)/Vivian/Tripadvisor hotel aspect-level sentiment analysis/'
              'data/data_processed/ABSA_restaurants_train.json') as f:
        absa_dataset = json.load(f)
    with open('/Users/qunyang/Dropbox (Personal)/Vivian/Tripadvisor hotel aspect-level sentiment analysis/'
              'data/data_processed/TripAdvisor_hotel_train.json') as f:
        tripadvisor_dataset = json.load(f)
    data = absa_dataset + tripadvisor_dataset
    data = json.dumps(data, indent=2)
    with open('/Users/qunyang/Dropbox (Personal)/Vivian/Tripadvisor hotel aspect-level sentiment analysis/'
              'data/data_processed/data.json', 'w') as f:
        f.write(data)
