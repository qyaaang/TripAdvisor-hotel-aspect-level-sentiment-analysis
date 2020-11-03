#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2020, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 29/10/20 2:48 PM
@describe:  
@version 1.0
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import argparse


class Resampling:

    def __init__(self, args):
        self.args = args
        self.data_pos = pd.read_excel('./data/data_origin/TripAdvisor_split_sample/'
                                      'TripAdvisor_hotel_positive.xlsx', header=None)
        self.data_neu = pd.read_excel('./data/data_origin/TripAdvisor_split_sample/'
                                      'TripAdvisor_hotel_neutral.xlsx', header=None)
        self.data_neg = pd.read_excel('./data/data_origin/TripAdvisor_split_sample/'
                                      'TripAdvisor_hotel_negative.xlsx', header=None)
        self.polarity_symbol = {'+': 'positive', '-': 'negative', '0': 'neutral', 'c': 'conflict'}

    def __call__(self, *args, **kwargs):
        self.create_dataset()
        data = pd.read_excel('./data/data_origin/TripAdvisor_hotel_{}_{}_{}_{}.xlsx'.
                             format(self.args.num_sample, self.args.frac_pos, self.args.frac_neu, self.args.frac_neg),
                             header=None)
        polarities = self.get_polarities(data)
        nums = self.sum_polarity(polarities)
        self.show_sample(nums)

    def resampling(self):
        """

        """
        data_pos = shuffle(self.data_pos).reset_index(drop=True)  # Shuffle dataset
        data_neu = shuffle(self.data_neu).reset_index(drop=True)
        data_neg = shuffle(self.data_neg).reset_index(drop=True)
        num_pos = int(self.args.num_sample * self.args.frac_pos)
        num_neu = int(self.args.num_sample * self.args.frac_neu)
        num_neg = int(self.args.num_sample * self.args.frac_neg)
        row1 = 0
        for i in range(len(data_pos)):
            data1 = data_pos[0:row1]
            polarities = self.get_polarities(data1)
            num_pos1, num_neu1, num_neg1 = self.sum_polarity(polarities)
            if num_pos1 >= num_pos:
                break
            row1 += 1
        row2 = 0
        for i in range(len(data_neu)):
            data2 = data_neu[0:row2]
            polarities = self.get_polarities(data2)
            num_pos2, num_neu2, num_neg2 = self.sum_polarity(polarities)
            if num_neu2 >= num_neu - num_neu1:
                break
            row2 += 1
        row3 = 0
        for i in range(len(data_neg)):
            data3 = data_neg[0:row3]
            polarities = self.get_polarities(data3)
            num_pos3, num_neu3, num_neg3 = self.sum_polarity(polarities)
            if num_neg3 >= num_neg - num_neg1 - num_neg2:
                break
            row3 += 1
        return data1, data2, data3

    def create_dataset(self):
        """

        """
        data_pos, data_neu, data_neg = self.resampling()
        data = pd.concat([data_pos, data_neu, data_neg], axis=0)
        data.to_excel('./data/data_origin/TripAdvisor_hotel_{}_{}_{}_{}.xlsx'.
                      format(self.args.num_sample, self.args.frac_pos, self.args.frac_neu, self.args.frac_neg),
                      index=None, header=None)

    def get_polarity(self, aspect_term):
        tmp = aspect_term.split(']')
        polarity = self.polarity_symbol[tmp[1].strip('[')]
        return polarity

    def get_polarities(self, data):
        polarities = []
        max_num_polarity = len(data.columns)
        for content_idx in range(len(data)):
            for polarity_idx in range(3, max_num_polarity):
                aspect_term = data.loc[content_idx, polarity_idx]
                if not pd.isna(aspect_term):
                    try:
                        polarity = self.get_polarity(aspect_term)
                        content = {'polarity': polarity}
                        polarities.append(content)
                    except:
                        continue
        return polarities

    def sum_polarity(self, polarities):
        """
        Summarize polarity
        """
        num_pos, num_neu, num_neg = 0, 0, 0
        for polarity in polarities:
            if polarity['polarity'] == 'positive':
                num_pos += 1
            elif polarity['polarity'] == 'neutral':
                num_neu += 1
            elif polarity['polarity'] == 'negative':
                num_neg += 1
        print('[Positive: {} Neutral: {} Negative: {}]\n'.
              format(num_pos, num_neu, num_neg))
        return num_pos, num_neu, num_neg

    def show_sample(self, nums):
        def size(pct, all_vals):
            absolute = int(pct / 100. * np.sum(all_vals))
            return '{:.1f}%\n({:d})'.format(pct, absolute)

        polarities = ['positive', 'neutral', 'negative']
        # nums = [self.num_pos, self.num_neu, self.num_neg]
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect='equal'))
        explode = (0, 0, 0.1)
        wedges, texts, autotexts = ax.pie(nums, autopct=lambda pct: size(pct, nums),
                                          explode=explode,
                                          textprops=dict(color='w'),
                                          shadow=False, startangle=90)
        ax.legend(wedges, polarities,
                  title='Polarities',
                  loc='upper right',
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=12, weight='bold')
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sample', default=3600, type=int)
    parser.add_argument('--frac_pos', default=0.4, type=float)
    parser.add_argument('--frac_neu', default=0.3, type=float)
    parser.add_argument('--frac_neg', default=0.3, type=float)
    plt.rcParams['font.family'] = 'Arial'
    args = parser.parse_args()
    resample = Resampling(args)
    resample()


if __name__ == '__main__':
    main()
