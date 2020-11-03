#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2020, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 10/10/20 8:05 PM
@describe:  
@version 1.0
"""


import matplotlib.pyplot as plt
import numpy as np
import argparse
import json


class Visualization:

    def __init__(self, args):
        self.args = args
        self.data = None
        plt.rcParams['font.family'] = 'Arial'
        self.font = {'family': 'Arial',
                     'style': 'normal',
                     'weight': 'bold',
                     'color': 'k',
                     }

    def file_name(self):
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
                                                            self.args.dataset,
                                                            self.args.num_sample,
                                                            self.args.frac_pos,
                                                            self.args.frac_neu,
                                                            self.args.frac_neg,
                                                            self.args.optimizer,
                                                            self.args.learning_rate,
                                                            self.args.weight_decay,
                                                            self.args.dropout,
                                                            self.args.batch_normalizations,
                                                            self.args.softmax)

    def get_data(self):
        file_name = './result/learning history/{}.json'.format(self.file_name())
        with open(file_name) as f:
            self.data = json.load(f)

    def plot_learning_curve(self):
        fig, ax = plt.subplots()
        x = np.arange(0, len(self.data['Loss']) + 1)
        ax.plot(x, np.array([0.0] + self.data['Training accuracy']) * 100)
        ax.scatter(x, np.array([0.0] + self.data['Training accuracy']) * 100, s=10, label='Training')
        ax.plot(x, np.array([0.0] + self.data['Validation accuracy']) * 100)
        ax.scatter(x, np.array([0.0] + self.data['Validation accuracy']) * 100, s=10, label='Validation')
        ax.set_xlim(0, self.args.num_epoch)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Epoch', fontdict=self.font)
        ax.set_ylabel('Accuracy (%)', fontdict=self.font)
        ax.legend(loc='lower right')
        plt.savefig('./result/figures/{}_learning_curve.png'.format(self.file_name()))
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='TD_LSTM', type=str)
    parser.add_argument('--dataset', default='TripAdvisor_hotel', type=str)
    parser.add_argument('--num_sample', default=3600, type=int)
    parser.add_argument('--frac_pos', default=0.4, type=float)
    parser.add_argument('--frac_neu', default=0.3, type=float)
    parser.add_argument('--frac_neg', default=0.3, type=float)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--batch_normalizations', action="store_true", default=False)
    parser.add_argument('--softmax', action="store_true", default=False)
    parser.add_argument('--dropout', default=0.50, type=float)
    args = parser.parse_args()
    studio = Visualization(args)
    studio.get_data()
    studio.plot_learning_curve()


if __name__ == '__main__':
    main()
