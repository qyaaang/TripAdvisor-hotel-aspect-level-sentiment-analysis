import torch
import sys
import numpy as np
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


class Test:

    def __init__(self, args):
        self.args = args
        tripadvisor_dataset = TripadvisorDatasetReader(dataset=args.dataset,
                                                       embed_dim=args.embed_dim,
                                                       max_seq_len=args.max_seq_len,
                                                       num_sample=args.num_sample,
                                                       frac_pos=args.frac_pos,
                                                       frac_neu=args.frac_neu,
                                                       frac_neg=args.frac_neg,
                                                       testset=args.testset
                                                       )
        self.test_data_loader = DataLoader(dataset=tripadvisor_dataset.test_data,
                                           batch_size=len(tripadvisor_dataset.test_data),
                                           shuffle=False)
        self.target_data_loader = DataLoader(dataset=tripadvisor_dataset.test_data,
                                             batch_size=len(tripadvisor_dataset.test_data),
                                             shuffle=False)
        self.mdl = args.model_class(self.args,
                                    embedding_matrix=tripadvisor_dataset.embedding_matrix,
                                    aspect_embedding_matrix=tripadvisor_dataset.aspect_embedding_matrix)
        self.test_info = {}

    def __call__(self, *args, **kwargs):
        self.test()

    @staticmethod
    def tensor_to_numpy(x):
        """
        Need to cast before calling numpy()
        """
        return x.data.type(torch.DoubleTensor).numpy()

    def evaluation(self, x):
        inputs = [x[col].to(device) for col in self.args.inputs_cols]
        targets = x['polarity'].to(device)
        outputs = self.mdl(inputs)
        outputs = self.tensor_to_numpy(outputs)
        targets = self.tensor_to_numpy(targets)
        outputs = np.argmax(outputs, axis=1)
        return outputs, targets

    def metric(self, targets, outputs, save_path=None):
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
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
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
                                                               self.args.softmax,
                                                               self.args.early_stopping
                                                               )

    def test(self):
        path = save_path + 'models/{}.model'.format(self.file_name())
        self.mdl.load_state_dict(torch.load(path))  # Load model
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
        self.test_info['Test accuracy'] = result['acc']
        self.test_info['Macro_precision'] = result['macro_precision']
        self.test_info['Macro_recall'] = result['macro_recall']
        self.test_info['Macro_F1'] = result['macro_f1']
        data = json.dumps(self.test_info, indent=2)
        with open('./result/learning history/test/{}_testset{}.json'.format(self.file_name(),
                                                                            self.args.testset), 'w') as f:
            f.write(data)
        # Plot confusion matrix
        class_names = ['Negative', 'Neutral', 'Positive']
        cnf_matrix = confusion_matrix(targets, outputs)
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix', normalize=True)
        # plt.show()
        plt.savefig('./result/figures/test/{}_testset{}.png'
                    .format(self.file_name(), self.args.testset)
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='TD_LSTM', type=str)
    parser.add_argument('--dataset', default='TripAdvisor_hotel', type=str)
    parser.add_argument('--num_sample', default=3600, type=int)
    parser.add_argument('--frac_pos', default=0.4, type=float)
    parser.add_argument('--frac_neu', default=0.3, type=float)
    parser.add_argument('--frac_neg', default=0.3, type=float)
    parser.add_argument('--testset', default='1', type=str)
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
    args.model_class = model_classes[args.model_name]
    args.inputs_cols = input_colses[args.model_name]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = Test(args)
    test()


if __name__ == '__main__':
    main()
