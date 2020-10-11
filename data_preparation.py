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


def data_preparation():
    modes = ['train', 'test']
    data_origin = pd.read_excel('{}/data/data_origin/TripAdvisor_hotel.xlsx'.format(base_path))
    train_set, test_set = train_test_split(data_origin, test_size=0.2, random_state=1993)
    train_set.to_csv('{}/data/data_origin/TripAdvisor_hotel_train.csv'.format(base_path), index=None)
    test_set.to_csv('{}/data/data_origin/TripAdvisor_hotel_test.csv'.format(base_path), index=None)
    for mode in modes:
        transformer = Transform(mode)
        transformer.get_data('{}/data/data_origin/TripAdvisor_hotel_{}.csv'.format(base_path, mode))
        transformer.write()
        transformer.write_json(base_path)


if __name__ == '__main__':
    data_preparation()
