#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2020, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 27/10/20 5:35 PM
@describe:  
@version 1.0
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class Hotel:

    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv('../{}.csv'.format(self.args.hotel_name),
                                header=None,
                                names=['Content', 'Rating'])
        self.rating_data = None
        self.rating_sys = {'high': {'cond': self.data['Rating'] >= 40,
                                    'file_name': 'high_rating'
                                    },
                           'mid': {'cond': self.data['Rating'] == 30,
                                   'file_name': 'mid_rating'
                                   },
                           'low': {'cond': self.data['Rating'] <= 20,
                                   'file_name': 'low_rating'
                                   },
                           }
        self.start_idx = {'Sheraton_Grand_Macao': {'high': 50000, 'mid': 500000, 'low': 5000000},
                          'Ritz_Carlton_Macau-Macau': {'high': 60000, 'mid': 600000, 'low': 6000000},
                          'Banyan_Tree_Macau': {'high': 80000, 'mid': 800000, 'low': 800000},
                          'JW_Marriott_Hotel_Macau': {'high': 90000, 'mid': 900000, 'low': 9000000},
                          'MGM_Macau-Macau': {'high': 150000, 'mid': 1500000, 'low': 15000000},
                          'Venetian_Macao_Resort_Hotel': {'high': 250000, 'mid': 2500000, 'low': 25000000},
                          'Conrad_Macao-Macau': {'high': 350000, 'mid': 3500000, 'low': 35000000},
                          'Grand_Hyatt_Macau-Macau': {'high': 450000, 'mid': 4500000, 'low': 45000000},
                          'St_Regis_Macao-Macau': {'high': 550000, 'mid': 5500000, 'low': 55000000},
                          'Wynn_Palace-Macau': {'high': 650000, 'mid': 6500000, 'low': 65000000}
                          }
        if not os.path.exists('./{}'.format(self.args.hotel_name)):
            os.mkdir('./{}'.format(self.args.hotel_name))
        if not os.path.exists('../Rating/{}'.format(self.args.hotel_name)):
            os.mkdir('../Rating/{}'.format(self.args.hotel_name))
        self.show_rating()
        self.split_rating()

    def __call__(self, *args, **kwargs):
        self.write_split_content()

    def show_rating(self):
        """
        Show rating distribution
        """
        def size(pct, all_vals):
            absolute = int(pct / 100. * np.sum(all_vals))
            return '{:.1f}%\n({:d})'.format(pct, absolute)
        ratings = list(self.rating_sys.keys())
        nums = [len(self.data[self.rating_sys[rating]['cond']]) for rating in ratings]
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect='equal'))
        explode = (0, 0, 0.1)
        wedges, texts, autotexts = ax.pie(nums, autopct=lambda pct: size(pct, nums),
                                          explode=explode,
                                          textprops=dict(color='w'),
                                          shadow=False, startangle=90)
        ax.legend(wedges, ratings,
                  title='Ratings',
                  loc='upper right',
                  bbox_to_anchor=(1, 0, 0.5, 1))
        ax.set_title('{}'.format(self.args.hotel_name))
        plt.setp(autotexts, size=12, weight='bold')
        plt.savefig('../Rating/{0}/{0}.png'.format(self.args.hotel_name))
        plt.show()

    def split_rating(self):
        """
        Split data according to ratings
        """
        self.rating_data = self.data[self.rating_sys[self.args.rating]['cond']]
        self.rating_data.to_csv('../Rating/{0}/{0}_{1}.csv'.
                                format(self.args.hotel_name,
                                       self.rating_sys[self.args.rating]['file_name']),
                                index=None, encoding='utf-8')

    @staticmethod
    def split_content(content):
        """
        Split content
        :param content:
        :return:
        """
        contents = []
        for c in content.split('.'):
            c_ = c.strip()
            contents.append(c_)
        while '' in contents:
            contents.remove('')
        return contents

    def write_split_content(self):
        """
        Write split content
        """
        _data = pd.DataFrame()
        start_idx = self.start_idx[self.args.hotel_name][self.args.rating]
        row = 0
        for content_idx, content in enumerate(self.rating_data['Content']):
            contents = self.split_content(content)
            for idx, _content in enumerate(contents):
                _data.loc[row, 0] = 'c{}'.format(start_idx + content_idx)
                _data.loc[row, 1] = 's{}'.format(idx + 1)
                _data.loc[row, 2] = _content
                row += 1
        _data.to_csv('./{0}/{0}_{1}_labeling.csv'.
                     format(self.args.hotel_name,
                            self.rating_sys[self.args.rating]['file_name']),
                     index=None, header=None, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hotel_name', default='Ritz_Carlton_Macau-Macau', type=str,
                        help='Sheraton_Grand_Macao, Ritz_Carlton_Macau-Macau, Banyan_Tree_Macau, '
                             'JW_Marriott_Hotel_Macau, MGM_Macau-Macau, Venetian_Macao_Resort_Hotel, '
                             'Conrad_Macao-Macau, Grand_Hyatt_Macau-Macau, St_Regis_Macao-Macau,'
                             'Wynn_Palace-Macau')
    parser.add_argument('--rating', default='high', type=str,
                        help='high, mid, low')
    args = parser.parse_args()
    plt.rcParams['font.family'] = 'Arial'
    hotel = Hotel(args)
    hotel()


if __name__ == '__main__':
    main()
