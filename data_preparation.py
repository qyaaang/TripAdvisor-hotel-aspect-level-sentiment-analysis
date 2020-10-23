#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2020, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 28/08/20 2:40 PM
@describe:  
@version 1.0
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing.transform import Transform

base_path = sys.path[0]


def data_preparation(dataset):
    modes = ['train', 'test']
    data_origin = pd.read_excel('{}/data/data_origin/{}.xlsx'.format(base_path, dataset))
    train_set, test_set = train_test_split(data_origin, test_size=0.2, random_state=1993)
    train_set.to_csv('{}/data/data_origin/{}_train.csv'.format(base_path, dataset), index=None)
    test_set.to_csv('{}/data/data_origin/{}_test.csv'.format(base_path, dataset), index=None)
    for mode in modes:
        transformer = Transform(mode)
        transformer.get_data('{}/data/data_origin/{}_{}.csv'.format(base_path, dataset, mode))
        transformer.write()
        transformer.write_json(base_path, dataset)


if __name__ == '__main__':
    dataset_name = 'TripAdvisor_hotel'
    # dataset_name = 'Sheraton_Grand_Macao'
    data_preparation(dataset_name)
