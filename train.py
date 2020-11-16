import torch
import torch.nn as nn
import torch.optim as optim
from adabelief_pytorch import AdaBelief
import sys
import time
import numpy as np
import random
from collections import Counter
from data_utils import TripadvisorDatasetReader
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from plot_utils import plot_confusion_matrix
import matplotlib.pyplot as plt
import argparse
import json

from models.ContextAvg import ContextAvg
from models.AE_ContextAvg import AEContextAvg
from models.LSTM import LSTM
from models.GRU import GRU
from models.CNN import CNN
from models.BiLSTM import BiLSTM
from models.BiGRU import BiGRU
from models.TD_LSTM import TD_LSTM
from models.TC_LSTM import TC_LSTM
from models.MemNet import MemNet
from models.IAN import IAN
from models.RAM import RAM
from models.AT_GRU import AT_GRU
from models.AT_LSTM import AT_LSTM
from models.AT_BiLSTM import AT_BiLSTM
from models.AT_BiGRU import AT_BiGRU
from models.ATAE_GRU import ATAE_GRU
from models.ATAE_LSTM import ATAE_LSTM
from models.ATAE_BiGRU import ATAE_BiGRU
from models.ATAE_BiLSTM import ATAE_BiLSTM
from models.LCRS import LCRS
from models.CABASC import CABASC
from models.GCAE import GCAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = sys.path[0] + '/result/'
base_path = sys.path[0] + '/data/store/'


def clip_gradient(parameters, clip):
    """
    Computes a gradient clipping coefficient based on gradient norm.
    """
    return nn.utils.clip_grad_norm(parameters, clip)


def tensor_to_numpy(x):
    """
    Need to cast before calling numpy()
    """
    return x.data.type(torch.DoubleTensor).numpy()


class BaseExperiment:
    """
    Implements a base experiment class for Aspect-Based Sentiment Analysis
    """

    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        if self.args.device == "cuda":
            torch.cuda.set_device(self.args.gpu)
            torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        print('> training arguments:')
        for arg in vars(args):
            print('>>> {0}: {1}'.format(arg, getattr(args, arg)))

        tripadvisor_dataset = TripadvisorDatasetReader(dataset=args.dataset,
                                                       embed_dim=args.embed_dim,
                                                       max_seq_len=args.max_seq_len,
                                                       num_sample=args.num_sample,
                                                       frac_pos=args.frac_pos,
                                                       frac_neu=args.frac_neu,
                                                       frac_neg=args.frac_neg,
                                                       testset=args.distribution
                                                       )
        if self.args.dev > 0.0:
            random.shuffle(tripadvisor_dataset.train_data.data)
            dev_num = int(len(tripadvisor_dataset.train_data.data) * self.args.dev)
            tripadvisor_dataset.dev_data.data = tripadvisor_dataset.train_data.data[:dev_num]
            tripadvisor_dataset.train_data.data = tripadvisor_dataset.train_data.data[dev_num:]
            # tripadvisor_dataset.train_data.data, tripadvisor_dataset.dev_data.data = \
            #     train_test_split(tripadvisor_dataset.train_data.data, test_size=self.args.dev, random_state=1993)

        # print(len(absa_dataset.train_data.data), len(absa_dataset.dev_data.data))

        self.train_data_loader = DataLoader(dataset=tripadvisor_dataset.train_data,
                                            batch_size=args.batch_size,
                                            shuffle=True)
        if self.args.dev > 0.0:
            self.dev_data_loader = DataLoader(dataset=tripadvisor_dataset.dev_data,
                                              batch_size=len(tripadvisor_dataset.dev_data),
                                              shuffle=False)
        self.test_data_loader = DataLoader(dataset=tripadvisor_dataset.test_data,
                                           batch_size=len(tripadvisor_dataset.test_data),
                                           shuffle=False)
        self.target_data_loader = DataLoader(dataset=tripadvisor_dataset.test_data,
                                             batch_size=len(tripadvisor_dataset.test_data),
                                             shuffle=False)
        self.mdl = args.model_class(self.args,
                                    embedding_matrix=tripadvisor_dataset.embedding_matrix,
                                    aspect_embedding_matrix=tripadvisor_dataset.aspect_embedding_matrix)
        self.reset_parameters()
        self.mdl.encoder.weight.requires_grad = True
        self.mdl.encoder_aspect.weight.requires_grad = True
        self.mdl.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_history = {}

    def reset_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.mdl.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    self.args.initializer(p)
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def select_optimizer(self):
        if self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.mdl.parameters()),
                                        lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'AdaBelief':
            self.optimizer = AdaBelief(self.mdl.parameters(),
                                       lr=self.args.learning_rate,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'RMS':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.mdl.parameters()),
                                           lr=self.args.learning_rate)
        elif self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.mdl.parameters()),
                                       lr=self.args.learning_rate,
                                       momentum=0.9,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.mdl.parameters()),
                                           lr=self.args.learning_rate)
        elif self.args.optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.mdl.parameters()),
                                            lr=self.args.learning_rate)

    def load_model(self, path):
        # mdl_best = self.load_model(PATH)
        # best_model_state = mdl_best.state_dict()/usr/bin/
        # model_state = self.mdl.state_dict()
        # best_model_state = {k: v for k, v in best_model_state.iteritems() if
        #                     k in model_state and v.size() == model_state[k].size()}
        # model_state.update(best_model_state)
        # self.mdl.load_state_dict(model_state)
        return torch.load(path)

    def train_batch(self, sample_batched):
        self.mdl.zero_grad()
        inputs = [sample_batched[col].to(device) for col in self.args.inputs_cols]
        targets = sample_batched['polarity'].to(device)
        outputs = self.mdl(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        clip_gradient(self.mdl.parameters(), 1.0)
        self.optimizer.step()
        # return loss.data[0]
        return loss.data

    def evaluation(self, x):
        inputs = [x[col].to(device) for col in self.args.inputs_cols]
        targets = x['polarity'].to(device)
        outputs = self.mdl(inputs)
        outputs = tensor_to_numpy(outputs)
        targets = tensor_to_numpy(targets)
        outputs = np.argmax(outputs, axis=1)
        return outputs, targets

    def metric(self, targets, outputs, save_path=None):
        dist = dict(Counter(outputs))
        acc = accuracy_score(targets, outputs)
        macro_recall = recall_score(targets, outputs, labels=[0, 1, 2], average='macro')
        macro_precision = precision_score(targets, outputs, labels=[0, 1, 2], average='macro')
        macro_f1 = f1_score(targets, outputs, labels=[0, 1, 2], average='macro')
        weighted_recall = recall_score(targets, outputs, labels=[0, 1, 2], average='weighted')
        weighted_precision = precision_score(targets, outputs, labels=[0, 1, 2], average='weighted')
        weighted_f1 = f1_score(targets, outputs, labels=[0, 1, 2], average='weighted')
        micro_recall = recall_score(targets, outputs, labels=[0, 1, 2], average='micro')
        micro_precision = precision_score(targets, outputs, labels=[0, 1, 2], average='micro')
        micro_f1 = f1_score(targets, outputs, labels=[0, 1, 2], average='micro')
        recall = recall_score(targets, outputs, labels=[0, 1, 2], average=None)
        precision = precision_score(targets, outputs, labels=[0, 1, 2], average=None)
        f1 = f1_score(targets, outputs, labels=[0, 1, 2], average=None)
        result = {'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1, 'macro_recall': macro_recall,
                  'macro_precision': macro_precision, 'macro_f1': macro_f1, 'micro_recall': micro_recall,
                  'micro_precision': micro_precision, 'micro_f1': micro_f1, 'weighted_recall': weighted_recall,
                  'weighted_precision': weighted_precision, 'weighted_f1': weighted_f1}
        # print("Output Distribution={}, Acc: {}, Macro-F1: {}".format(dist, acc, macro_f1))
        if save_path is not None:
            f_to = open(save_path, 'w')
            f_to.write("lr: {}\n".format(self.args.learning_rate))
            f_to.write("batch_size: {}\n".format(self.args.batch_size))
            f_to.write("opt: {}\n".format(self.args.optimizer))
            f_to.write("max_sentence_len: {}\n".format(self.args.max_seq_len))
            f_to.write("end params -----------------------------------------------------------------\n")
            for key in result.keys():
                f_to.write("{}: {}\n".format(key, result[key]))
            f_to.write("end metrics -----------------------------------------------------------------\n")
            for i in range(len(outputs)):
                f_to.write("{}: {},{}\n".format(i, outputs[i], targets[i]))
            f_to.write("end ans -----------------------------------------------------------------\n")
            f_to.close()
        return result

    def file_name(self):
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
                                                            self.args.dataset,
                                                            self.args.num_sample,
                                                            self.args.frac_pos,
                                                            self.args.frac_neu,
                                                            self.args.frac_neg,
                                                            self.args.optimizer,
                                                            self.args.learning_rate,
                                                            self.args.num_epoch,
                                                            self.args.dropout,
                                                            self.args.batch_normalizations,
                                                            self.args.softmax)

    def validation(self):
        self.mdl.eval()
        val_data_loader = self.dev_data_loader if self.args.dev > 0.0 else self.test_data_loader
        outputs, targets = None, None
        with torch.no_grad():
            for v_batch, v_sample_batched in enumerate(val_data_loader):
                output, target = self.evaluation(v_sample_batched)
                outputs = output if outputs is None else np.concatenate((outputs, output))
                targets = target if targets is None else np.concatenate((targets, target))
            result = self.metric(targets=targets, outputs=outputs)
        return result

    def validation_early_stopping(self):
        """
        Early stopping to prevent overfitting
        """
        self.mdl.eval()
        val_data_loader = self.dev_data_loader if self.args.dev > 0.0 else self.test_data_loader
        outputs, targets = None, None
        losses = []
        for v_batch, v_sample_batched in enumerate(val_data_loader):
            output, target = self.evaluation(v_sample_batched)
            loss = self.train_batch(v_sample_batched)
            losses.append(loss)
            outputs = output if outputs is None else np.concatenate((outputs, output))
            targets = target if targets is None else np.concatenate((targets, target))
        result = self.metric(targets=targets, outputs=outputs)
        avg_loss = np.mean(losses)
        return result, avg_loss

    def train(self):
        best_acc = 0.0
        best_epoch = 1
        i, j = 0, 0
        p = 10
        best_loss = 10000000
        best_result = None
        self.select_optimizer()
        losses_train = []
        accuracy_train, accuracy_validation = [], []
        for epoch in range(self.args.num_epoch):
            losses = []
            self.mdl.train()
            t0 = time.time()
            outputs_train, targets_train = None, None
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                loss = self.train_batch(sample_batched)
                losses.append(loss)
                output_train, target_train = self.evaluation(sample_batched)
                outputs_train = output_train if outputs_train is None else np.concatenate((outputs_train, output_train))
                targets_train = target_train if targets_train is None else np.concatenate((targets_train, target_train))
            results_train = self.metric(targets=targets_train, outputs=outputs_train)
            t1 = time.time()
            # Validation
            if not self.args.early_stopping:
                result = self.validation()
                if result['acc'] > best_acc:
                    best_acc = result['acc']
                    path = save_path + 'models/{}.model'.format(self.file_name())
                    torch.save(self.mdl.state_dict(), path)
                    best_result = result
                    best_epoch = epoch + 1
                print('\033[1;31m[Epoch {:>4}]\033[0m  '
                      '\033[1;31mTraining loss={:.5f}\033[0m  '
                      '\033[1;32mTraining accuracy={:.2f}%\033[0m  '
                      '\033[1;33mValidation accuracy={:.2f}%\033[0m  '
                      'Time cost={:.2f}s'.format(epoch + 1,
                                                 np.mean(losses),
                                                 results_train['acc'] * 100,
                                                 result['acc'] * 100,
                                                 t1 - t0))
            else:  # Early stopping
                result, val_loss = self.validation_early_stopping()
                if val_loss < best_loss:
                    j = 0
                    path = save_path + 'models/{}.model'.format(self.file_name())
                    torch.save(self.mdl.state_dict(), path)
                    best_loss = val_loss
                    best_result = result
                    best_epoch = epoch + 1
                else:
                    j += 1
                print('\033[1;31m[Epoch {:>4}]\033[0m  '
                      '\033[1;31mTraining loss={:.5f}\033[0m  '
                      '\033[1;32mTraining accuracy={:.2f}%\033[0m  '
                      '\033[1;31mValidation loss={:.5f}\033[0m  '
                      '\033[1;33mValidation accuracy={:.2f}%\033[0m  '
                      'Time cost={:.2f}s'.format(epoch + 1,
                                                 np.mean(losses),
                                                 results_train['acc'] * 100,
                                                 val_loss,
                                                 result['acc'] * 100,
                                                 t1 - t0))
                if j >= p:
                    break
            losses_train.append(np.mean(losses))
            accuracy_train.append(results_train['acc'])
            accuracy_validation.append(result['acc'])
        self.learning_history['Loss'] = np.array(losses_train).tolist()
        self.learning_history['Training accuracy'] = np.array(accuracy_train).tolist()
        self.learning_history['Validation accuracy'] = np.array(accuracy_validation).tolist()
        self.learning_history['Best Validation accuracy'] = best_result['acc']
        self.learning_history['Best Validation epoch'] = best_epoch

    def test(self, frac_pos, frac_neu, frac_neg):
        path = save_path + 'models/{}.model'.format(self.file_name())
        self.mdl.load_state_dict(self.load_model(path))  # Load model
        self.mdl.eval()
        outputs, targets = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                output, target = self.evaluation(t_sample_batched)
                outputs = output if outputs is None else np.concatenate((outputs, output))
                targets = target if targets is None else np.concatenate((targets, target))
        result = self.metric(targets=targets, outputs=outputs)
        print('\033[1;32mTest accuracy:{:.2f}%\nrecall: {}\nprecision: {}\nmacro-F1:{:.2f}%\033[0m'.
              format(result['acc'] * 100,
                     result['recall'],
                     result['precision'],
                     result['macro_f1'] * 100))
        self.learning_history['Test accuracy'] = result['acc']
        self.learning_history['Macro_precision'] = result['macro_precision']
        self.learning_history['Macro_recall'] = result['macro_recall']
        self.learning_history['Macro_F1'] = result['macro_f1']
        # Plot confusion matrix
        class_names = ['Negative', 'Neutral', 'Positive']
        cnf_matrix = confusion_matrix(targets, outputs)
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix', normalize=True)
        # plt.show()
        plt.savefig('./result/figures/'
                    '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png'
                    .format(self.args.model_name,
                            self.args.dataset,
                            self.args.num_sample,
                            str(frac_pos),
                            str(frac_neu),
                            str(frac_neg),
                            self.args.optimizer,
                            self.args.learning_rate,
                            self.args.num_epoch,
                            self.args.dropout,
                            self.args.batch_normalizations,
                            self.args.softmax))

    def write_learning_history(self):
        data = json.dumps(self.learning_history, indent=2)
        with open('./result/learning history/{}.json'.format(self.file_name()), 'w') as f:
            f.write(data)

    def transfer_learning(self):
        model_path = save_path + 'models/{}_{}_{}_{}_{}_{}_{}_{}_{}.model'.format(self.args.model_name,
                                                                                  self.args.pre_trained_model,
                                                                                  self.args.optimizer,
                                                                                  self.args.learning_rate,
                                                                                  self.args.max_seq_len,
                                                                                  self.args.dropout,
                                                                                  self.args.softmax,
                                                                                  self.args.batch_size,
                                                                                  self.args.dev)
        self.mdl.load_state_dict(self.load_model(model_path))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='TD_LSTM', type=str)
    parser.add_argument('--dataset', default='TripAdvisor_hotel', type=str)
    parser.add_argument('--num_sample', default=3600, type=int)
    parser.add_argument('--frac_pos', default=0.4, type=float)
    parser.add_argument('--frac_neu', default=0.3, type=float)
    parser.add_argument('--frac_neg', default=0.3, type=float)
    parser.add_argument('--distribution', default='1', type=str)
    parser.add_argument('--pre_trained_model', default='ABSA', type=str)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--kernel_num', default=100, type=int)
    parser.add_argument('--kernel_sizes', default=[3, 4, 5], nargs='+', type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--batch_normalizations', action='store_true', default=False)
    parser.add_argument('--softmax', action='store_true', default=False)
    parser.add_argument('--dev', default=0.20, type=float)
    parser.add_argument('--dropout', default=0.50, type=float)
    args = parser.parse_args()
    model_classes = {
        'ContextAvg': ContextAvg,
        'AEContextAvg': AEContextAvg,
        'CNN': CNN,
        'TD_LSTM': TD_LSTM,
        'LSTM': LSTM,
        'GRU': GRU,
        'BiGRU': BiGRU,
        'BiLSTM': BiLSTM,
        'MemNet': MemNet,
        'IAN': IAN,
        'RAM': RAM,
        'AT_GRU': AT_GRU,
        'AT_LSTM': AT_LSTM,
        'AT_BiGRU': AT_BiGRU,
        'AT_BiLSTM': AT_BiLSTM,
        'ATAE_GRU': ATAE_GRU,
        'ATAE_LSTM': ATAE_LSTM,
        'ATAE_BiGRU': ATAE_BiGRU,
        'ATAE_BiLSTM': ATAE_BiLSTM,
        'TC_LSTM': TC_LSTM,
        'LCRS': LCRS,
        'CABASC': CABASC,
        'GCAE': GCAE
    }
    input_colses = {
        'LSTM': ['text_raw_indices'],
        'CNN': ['text_raw_indices'],
        'GRU': ['text_raw_indices'],
        'BiGRU': ['text_raw_indices'],
        'BiLSTM': ['text_raw_indices'],
        'ContextAvg': ['text_raw_indices'],
        'AT_GRU': ['text_raw_indices'],
        'AT_LSTM': ['text_raw_indices'],
        'AT_BiGRU': ['text_raw_indices'],
        'AT_BiLSTM': ['text_raw_indices'],
        'ATAE_GRU': ['text_raw_indices', 'aspect_indices'],
        'ATAE_LSTM': ['text_raw_indices', 'aspect_indices'],
        'ATAE_BiGRU': ['text_raw_indices', 'aspect_indices'],
        'ATAE_BiLSTM': ['text_raw_indices', 'aspect_indices'],
        'AEContextAvg': ['text_raw_indices', 'aspect_indices'],
        'TD_LSTM': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'TC_LSTM': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices', 'aspect_indices'],
        'IAN': ['text_raw_indices', 'aspect_indices'],
        'MemNet': ['text_raw_without_aspect_indices', 'aspect_indices', 'text_left_with_aspect_indices'],
        'RAM': ['text_raw_indices', 'aspect_indices'],
        'CABASC': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
                   'text_right_with_aspect_indices'],
        'LCRS': ['text_left_indices', 'aspect_indices', 'text_right_indices'],
        'GCAE': ['text_raw_indices', 'aspect_indices']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
        'kaiming_normal_': torch.nn.init.kaiming_normal_
    }
    distributions = {'1': {'num': 4000, 'frac_pos': 0.35, 'frac_neu': 0.35, 'frac_neg': 0.3},
                     '2': {'num': 4000, 'frac_pos': 0.6, 'frac_neu': 0.15, 'frac_neg': 0.25},
                     '3': {'num': 4000, 'frac_pos': 0.25, 'frac_neu': 0.6, 'frac_neg': 0.15},
                     '4': {'num': 4000, 'frac_pos': 0.25, 'frac_neu': 0.15, 'frac_neg': 0.6},
                     }
    args.model_class = model_classes[args.model_name]
    args.inputs_cols = input_colses[args.model_name]
    args.initializer = initializers[args.initializer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.batch_normalizations = False
    exp = BaseExperiment(args)
    exp.train()
    exp.test(frac_pos=distributions[args.distribution]['frac_pos'],
             frac_neu=distributions[args.distribution]['frac_neu'],
             frac_neg=distributions[args.distribution]['frac_neg']
             )
    exp.write_learning_history()


if __name__ == '__main__':
    main()
