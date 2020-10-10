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

    def get_data(self):
        file_name = './result/learning history/{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(self.args.model_name,
                                                                                       self.args.dataset,
                                                                                       self.args.optimizer,
                                                                                       self.args.learning_rate,
                                                                                       self.args.max_seq_len,
                                                                                       self.args.dropout,
                                                                                       self.args.softmax,
                                                                                       self.args.batch_size,
                                                                                       self.args.dev)
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
        plt.savefig('./result/figures/{}_{}_{}_{}_{}_{}_{}_{}_{}_learning_curve.png'.format(self.args.model_name,
                                                                                            self.args.dataset,
                                                                                            self.args.optimizer,
                                                                                            self.args.learning_rate,
                                                                                            self.args.max_seq_len,
                                                                                            self.args.dropout,
                                                                                            self.args.softmax,
                                                                                            self.args.batch_size,
                                                                                            self.args.dev))
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='TD_LSTM', type=str)
    parser.add_argument('--dataset', default='TripAdvisor hotel', type=str)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--softmax', action="store_true", default=False)
    parser.add_argument('--dev', default=0.20, type=float)
    parser.add_argument('--dropout', default=0.50, type=float)
    args = parser.parse_args()
    studio = Visualization(args)
    studio.get_data()
    studio.plot_learning_curve()


if __name__ == '__main__':
    main()
