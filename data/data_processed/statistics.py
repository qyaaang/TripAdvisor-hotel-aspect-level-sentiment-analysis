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
import numpy as np
import matplotlib.pyplot as plt


class Sample:

    def __init__(self, dataset, train):
        self.train = train
        self.file_name = '{}_{}.json'.format(dataset, train)
        with open(self.file_name) as f:
            self.data = json.load(f)
        self.num_pos, self.num_neu, self.num_neg = 0, 0, 0

    def __call__(self):
        self.sum_polarity()
        data = self.write_data()
        return data

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
        print('{} [Positive: {} Neutral: {} Negative: {}]\n'.
              format(self.train, self.num_pos, self.num_neu, self.num_neg))

    def write_data(self):
        data = {'{} set'.format(self.train): {}}
        data['{} set'.format(self.train)]['positive'] = self.num_pos
        data['{} set'.format(self.train)]['neutral'] = self.num_neu
        data['{} set'.format(self.train)]['negative'] = self.num_neg
        return data


def sample_distribution(dataset):
    modes = ['train', 'test']
    polarities = ['positive', 'neutral', 'negative']
    data = {}
    for mode in modes:
        sample = Sample(dataset, mode)
        data.update(sample())
    data['dataset'] = {}
    for polarity in polarities:
        data['dataset'][polarity] = data['train set'][polarity] + data['test set'][polarity]
    return data


def bar_distribution(dataset, is_int=True):
    data = sample_distribution(dataset)
    polarities = list(data['dataset'].keys())
    nums = [data['dataset'][polarity] for polarity in polarities]
    fig, ax = plt.subplots()
    font = {'family': 'Arial',
            'style': 'normal',
            'weight': 'bold',
            'color': 'k',
            }
    x = range(len(polarities))
    bars = ax.bar(x, height=nums, width=0.4, alpha=0.8)
    plt.xticks([idx for idx in x], polarities)
    ax.set_ylabel('Number', fontdict=font)
    for bar in bars:
        height = int(bar.get_height()) if is_int else bar.get_height()
        ax.text(bar.get_x() + bar.get_height() / 2, height + 0.1,
                str(height))
    plt.show()


def pie_distribution(dataset):

    def size(pct, all_vals):
        absolute = int(pct / 100. * np.sum(all_vals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    data = sample_distribution(dataset)
    polarities = list(data['dataset'].keys())
    nums = [data['dataset'][polarity] for polarity in polarities]
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))
    explode = (0, 0, 0.1)
    wedges, texts, autotexts = ax.pie(nums, autopct=lambda pct: size(pct, nums),
                                      explode=explode,
                                      textprops=dict(color="w"),
                                      shadow=False, startangle=90)
    ax.legend(wedges, polarities,
              title='Polarities',
              loc='upper right',
              bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=12, weight="bold")
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    dataset_name = 'TripAdvisor_hotel'
    # dataset_name = 'Sheraton_Grand_Macao'
    # bar_distribution()
    pie_distribution(dataset_name)
