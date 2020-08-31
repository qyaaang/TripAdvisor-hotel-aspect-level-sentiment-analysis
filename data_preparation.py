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
from data_processing.transform import Transform
from data_processing.absa_dataset import ABSA

base_path = sys.path[0]
data_origin_path = base_path + '/data/data_origin/hotel_test.xlsx'
data_processed_path = base_path + '/data/data_processed/hotel_test.json'

transformer = Transform()

transformer.get_data(data_origin_path)

data = transformer.data_origin

# text_id = transformer.get_id(114)
# text = transformer.get_text(114)
#
# transformer.get_aspect_term(114)

transformer.write()
transformer.write_json(data_processed_path)

# data_path = base_path + '/data/data_processed/ABSA Dataset/'
# save_path = base_path + '/data/data_processed/'
# dataset_list = ['train', 'test']
# for dataset in dataset_list:
#     absa = ABSA()
#     absa.set_dataset_name(['restaurants14', 'restaurants15', 'restaurants16'])
#     absa.pack_data(data_path, dataset)
#     absa.write_json(save_path, dataset)

