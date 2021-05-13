import copy
import os
import re
import pgl
import collections
import pickle
import numpy as np
import paddle
from const import PAD
from paddle.io import Dataset
from tqdm import tqdm
import pahelix.toolkit.linear_rna as linear_rna


class Vocabulary(object):
    def __init__(self):
        self.frozen = False
        self.values = []  # words_dict
        self.indices = {}  # word2index
        self.counts = collections.defaultdict(int)  # 统计每种words的数目

    @property
    def size(self):
        return len(self.values)

    def value(self, index):
        assert 0 <= index < len(self.values)
        return self.values[index]

    def index(self, value):  # 根据word索引出index
        if not self.frozen:
            self.counts[value] += 1

        if value in self.indices:
            return self.indices[value]

        elif not self.frozen:
            self.values.append(value)
            self.indices[value] = len(self.values) - 1
            return self.indices[value]
        else:
            raise ValueError("Unknown value: {}".format(value))

    def count(self, value):  # 返回指定word的数目
        return self.counts[value]

    def freeze(self):  # 冻结 words_dict和word2index，不再更新
        self.frozen = True


def data_add(seq, random=(10, 20, 30)):
    dot_aug = []
    for r in random:
        dot = linear_rna.linear_fold_v(seq, beam_size=r, no_sharp_turn=True)[0]
        dot_aug.append(dot)
    return dot_aug


def read_data(filename, test=False, data_aug=False, reload=False):  # return list[dict{id,sequence,structure,label}]
    if data_aug:
        print("Data Augmentation with linear_rna!")
    if not os.path.exists("data/raw_data"):
        os.mkdir("data/raw_data")

    data_type = os.path.split(filename)[-1].split('.')[0]
    if data_aug:
        save_path = 'data/raw_data/{}_aug.pkl'.format(data_type)
    else:
        save_path = 'data/raw_data/{}.pkl'.format(data_type)

    if os.path.exists(save_path) and not reload:
        data = pickle.load(open(save_path, 'rb'))
    else:
        data, x = [], []
        w = open(filename, "r").readlines()
        for index in tqdm(range(len(w))):
            line = w[index].strip()
            if not line:  # 遇到空行则处理前面待处理的数据
                ID, seq, dot = x[:3]
                if test:
                    if data_aug:
                        x = {"id": ID,
                             "sequence": seq,
                             "structure": dot,
                             "dot_aug": data_add(seq)  # add
                             }
                    else:
                        x = {"id": ID,
                             "sequence": seq,
                             "structure": dot,
                             }

                    data.append(x)
                    x = []
                    continue

                punp = x[3:]
                punp = [punp_line.split() for punp_line in punp]
                punp = [(float(p)) for i, p in punp]
                if data_aug:
                    x = {"id": ID,
                         "sequence": seq,
                         "structure": dot,
                         "p_unpaired": punp,
                         "dot_aug": data_add(seq)  # add
                         }
                else:
                    x = {"id": ID,
                         "sequence": seq,
                         "structure": dot,
                         "p_unpaired": punp,
                         }

                data.append(x)
                x = []
            else:  # 非空行则加入待处理队列
                x.append(line)

        pickle.dump(data, open(save_path, 'wb'))

    return data


def load_train_data(data_aug=False):
    assert os.path.exists("data/train.txt")
    assert os.path.exists("data/dev.txt")
    train = read_data("data/train.txt", data_aug=data_aug)
    dev = read_data("data/dev.txt", data_aug=data_aug)
    return train, dev


def load_test_data(name, data_aug=False):
    if name == "a":
        assert os.path.exists("data/test_nolabel.txt")
        test = read_data("data/test_nolabel.txt", test=True, data_aug=data_aug)
    elif name == "b":
        assert os.path.exists("data/B_board_112_seqs.txt")
        test = read_data("data/B_board_112_seqs.txt", test=True, data_aug=data_aug)
    else:
        raise ValueError("name must in [a, b]")
    return test


def load_test_label_data():
    assert os.path.exists("data/test.txt")
    test = read_data("data/test.txt")
    return test


def create_vocab(length, base):
    pad_num = (length - 1) // 2
    pattern = re.compile('(_[A-Z]+_)|(=[A-Z]+=)|([A-Z]=+[A-Z])|([A-Z]_+[A-Z])')

    def recur(length, token=''):
        if len(token) == length:

            if all(['=' not in token[: pad_num], '_' not in token[length - pad_num:],
                    '=' not in token[pad_num: -pad_num], '_' not in token[pad_num: -pad_num],
                    re.search(pattern, token) is None]):
                yield token
        else:
            for each in base:
                for j in recur(length, token + each):
                    yield j

    for each in recur(length, ''):
        yield each


def get_pair(texts):
    '''
    :param texts: list[str]
    :return: list[list[tuple]]
    '''
    assert isinstance(texts, list) and isinstance(texts[0], str)
    all_pairs = []
    for text in texts:
        stack = []
        pair = []
        for index, char in enumerate(text):
            if char == '(':
                stack.append(index)  # push the left index
            elif char == ')':
                assert len(stack) != 0
                left_index = stack[-1]  # pop the left index
                stack.pop()
                pair.append((left_index, index))

        # pair = pair[::-1]  # reverse
        pair = sorted(pair, key=lambda x: x[0], reverse=False)
        all_pairs.append(pair)

    return all_pairs


def process_vocabulary(n_gram=3):
    """
    Creates and returns vocabulary objects.
    Only iterates through the first 100 sequences, to save computation.
    """
    if n_gram % 2 == 0:
        raise ValueError("n should be a single int.")

    seq_vocab = Vocabulary()
    bracket_vocab = Vocabulary()

    for vocab in [seq_vocab, bracket_vocab]:
        vocab.index(PAD)

    seq_base = ["_", "A", "C", "G", "U", "="]
    dot_base = ["_", ".", "(", ")", "="]
    seq = list(create_vocab(n_gram, seq_base))
    dot = list(create_vocab(n_gram, dot_base))
    for character in seq:
        seq_vocab.index(character)
    for character in dot:
        bracket_vocab.index(character)

    for vocab in [seq_vocab, bracket_vocab]:
        vocab.freeze()

    return seq_vocab, bracket_vocab


def rna2graph(sequence, pair, dot, encode_bond_type=False):
    edges = []
    edges_attr = []
    # add neighbour
    for i in range(len(sequence) - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    edges_attr += [0] * 2 * (len(sequence) - 1)

    # add pair
    for each in pair:
        edges.append(each)
        edges.append((each[1], each[0]))
        if {sequence[each[0]], sequence[each[1]]} == {'A', 'U'}:
            edges_attr += [1] * 2
        elif {sequence[each[0]], sequence[each[1]]} == {'C', 'G'}:
            edges_attr += [2] * 2
        elif {sequence[each[0]], sequence[each[1]]} == {'G', 'U'}:
            edges_attr += [3] * 2
        else:
            raise ValueError('Wrong Pair!')
    # add self loop
    for i in range(len(sequence)):
        edges.append((i, i))
    edges_attr += [4] * len(sequence)

    num_nodes = len(sequence)
    edges_attr = np.array(edges_attr).reshape(-1, 1)
    # edges_attr = np.array([0] * 2 * (len(sequence) - 1) + [1] * (2 * len(pair)) + [2] * len(sequence)).reshape(-1, 1)
    seq_dic = {k: v for v, k in enumerate(['A', 'C', 'G', 'U'])}
    dot_dic = {k: v for v, k in enumerate(['.', '(', ')'])}
    node_attr = np.array([seq_dic[c] for c in sequence]).reshape(-1, 1)
    dot_attr = np.array([dot_dic[c] for c in dot]).reshape(-1, 1)
    graph = pgl.graph.Graph(edges=edges, num_nodes=num_nodes, node_feat={'one_hot_seq_feature': node_attr,
                                                                         'one_hot_dot_feature': dot_attr},
                            edge_feat={'one_hot_feature': edges_attr})
    return graph


class MyDataset(Dataset):
    def __init__(self, data_label, data_type, seq_vocab, bracket_vocab, n_gram, process=False,
                 graph=False, bond=False, data_aug=False):
        super(MyDataset, self).__init__()
        assert isinstance(data_type, str) and data_type in ['train', 'val', 'test', 'test_b']
        assert n_gram % 2 != 0
        self.data = None
        self.data_type = data_type
        self.n_gram = n_gram
        if not os.path.exists('data/ready_data'):
            os.makedirs('data/ready_data')
        if not graph:
            if not data_aug:
                data_path = os.path.join('data/ready_data', f"{n_gram}_gram_{data_type}.pkl")
            else:
                data_path = os.path.join('data/ready_data', f"{n_gram}_gram_{data_type}_aug.pkl")
        else:
            data_path = os.path.join('data/ready_data', f"{data_type}_{bond}_graph.pkl")
        print('Loading the {} data'.format(data_type))
        if os.path.exists(data_path) and not process:
            self.data = pickle.load(open(data_path, 'rb'))
        else:
            print('process data....')
            all_data = []
            if not graph:
                for d in data_label:
                    id = d['id']
                    seq = self.word2index(d['sequence'], seq_vocab)
                    struct = self.word2index(d['structure'], bracket_vocab)
                    dot_aug = [self.word2index(each, bracket_vocab) for each in d['dot_aug']]
                    dot_aug.append(struct)
                    struct = dot_aug
                    length = len(seq)
                    if data_type not in ['test', 'test_b']:
                        label = d['p_unpaired']
                        all_data.append({'id': id, 'seq': seq, 'dot': struct, 'label': label, 'length': length})
                    else:
                        all_data.append({'id': id, 'seq': seq, 'dot': struct, 'length': length})
            else:
                seq_list = [each['sequence'] for each in data_label]
                dot_list = [each['structure'] for each in data_label]
                pair_list = get_pair(dot_list)
                graph_list = [rna2graph(sequence, pair, dot) for sequence, pair, dot in
                              zip(seq_list, pair_list, dot_list)]
                if data_type != 'test':
                    label_list = [np.array(each['p_unpaired'], dtype=np.float32) for each in data_label]
                    for graph, label in zip(graph_list, label_list):
                        all_data.append({'graph': graph, 'label': label})
                else:
                    for graph in graph_list:
                        all_data.append({'graph': graph})
            self.data = all_data
            pickle.dump(all_data, open(data_path, 'wb'))

        print('\tnumbers of {}: {}'.format(data_type, self.__len__()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def word2index(self, text, vocab):
        pad_num = (self.n_gram - 1) // 2
        text = '_' * pad_num + text + '=' * pad_num
        index_list = [vocab.index(text[i: i + self.n_gram]) for i in range(len(text) - self.n_gram + 1)]
        return index_list

    @property
    def max_length(self):
        return max([d['length'] for d in self.data])


class Old_MyDataset(Dataset):
    def __init__(self, data_label, data_type, seq_vocab, bracket_vocab, n_gram, process=False):
        super(MyDataset, self).__init__()
        assert isinstance(data_type, str) and data_type in ['train', 'val', 'test']
        assert n_gram % 2 != 0
        self.data = None
        self.data_type = data_type
        self.n_gram = n_gram
        if not os.path.exists('data/ready_data'):
            os.makedirs('data/ready_data')
        data_path = os.path.join('data/ready_data', f"{n_gram}_gram_{data_type}.pkl")
        print('Loading the {} data'.format(data_type))
        if os.path.exists(data_path) and not process:
            self.data = pickle.load(open(data_path, 'rb'))
        else:
            print('process data....')
            all_data = []
            for d in data_label:
                id = d['id']
                seq = self.word2index(d['sequence'], seq_vocab)
                struct = self.word2index(d['structure'], bracket_vocab)
                length = len(seq)
                if data_type != 'test':
                    label = d['p_unpaired']
                    all_data.append({'id': id, 'seq': seq, 'dot': struct, 'label': label, 'length': length})
                else:
                    all_data.append({'id': id, 'seq': seq, 'dot': struct, 'length': length})

            self.data = all_data
            pickle.dump(all_data, open(data_path, 'wb'))

        print('\tnumbers of {}: {}'.format(data_type, self.__len__()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def word2index(self, text, vocab):
        pad_num = (self.n_gram - 1) // 2
        text = '_' * pad_num + text + '=' * pad_num
        index_list = [vocab.index(text[i: i + self.n_gram]) for i in range(len(text) - self.n_gram + 1)]
        return index_list

    @property
    def max_length(self):
        return max([d['length'] for d in self.data])


class PretrainingDataset(Dataset):
    """
    Create a dataset for pretraining data(data in which has already been randomly masked)
    """

    def __init__(self, input_file, max_pred_length):
        super().__init__()
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.inputs = pickle.load(open(input_file, 'rb'))
        # self.keys = ['input_seq_ids', 'input_seq_mask', 'input_dot_ids', 'input_dot_mask',
        #              'masked_seq_positions', 'masked_seq_ids', 'masked_dot_positions', 'masked_dot_ids']
        self.keys = ['input_seq_ids', 'input_dot_ids', 'input_mask',
                     'masked_seq_positions', 'masked_seq_ids', 'masked_dot_positions', 'masked_dot_ids']
        # self.inputs = [np.asarray(f[key][:]) for key in keys]
        # f.close()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs['input_seq_ids'])

    def __getitem__(self, index):
        """
        Return a list: [input_seq_ids, input_dot_ids, input_mask,
                        masked_seq_labels, masked_dot_labels]
        """
        data = {key: self.inputs[key][index] for key in self.keys}

        return data


class PretrainNormCollator:
    def __init__(self, max_pred_length):
        # self.keys = ['input_seq_ids', 'input_seq_mask', 'input_dot_ids', 'input_dot_mask',
        #              'masked_seq_positions', 'masked_seq_ids', 'masked_dot_positions', 'masked_dot_ids']
        self.keys = ['input_seq_ids', 'input_dot_ids', 'input_mask',
                     'masked_seq_positions', 'masked_seq_ids', 'masked_dot_positions', 'masked_dot_ids']
        self.max_pred_length = max_pred_length

    def __call__(self, examples):
        examples = copy.deepcopy(examples)
        # [input_seq_ids, input_seq_mask, input_dot_ids, input_dot_mask, masked_seq_positions, masked_seq_ids,
        #  masked_dot_positions, masked_dot_ids] = [self.unpack_field(examples, key) for key in self.keys]
        [input_seq_ids, input_dot_ids, input_mask, masked_seq_positions, masked_seq_ids,
         masked_dot_positions, masked_dot_ids] = [self.unpack_field(examples, key) for key in self.keys]

        max_seq_length = max([len(each) for each in input_seq_ids])
        # print("MAX", max_seq_length)

        # pad sequence with 0
        # for seq_id, seq_mask in zip(input_seq_ids, input_seq_mask):
        #     seq_id.extend([0] * (max_seq_length - len(seq_id)))
        #     seq_mask.extend([0] * (max_seq_length - len(seq_mask)))
        #
        # for dot_id, dot_mask in zip(input_dot_ids, input_dot_mask):
        #     dot_id.extend([0] * (max_seq_length - len(dot_id)))
        #     dot_mask.extend([0] * (max_seq_length - len(dot_mask)))
        for seq_id, dot_id, mask in zip(input_seq_ids, input_dot_ids, input_mask):
            seq_id.extend([0] * (max_seq_length - len(seq_id)))
            dot_id.extend([0] * (max_seq_length - len(dot_id)))
            mask.extend([0] * (max_seq_length - len(mask)))

        # assert len(input_seq_ids[0]) == len(input_dot_ids[0]) == len(input_seq_mask[0]) == len(input_dot_mask[0]) == max_seq_length
        assert len(input_seq_ids[0]) == len(input_dot_ids[0]) == len(input_mask[0]) == max_seq_length

        # [input_seq_ids, input_seq_mask, input_dot_ids, input_dot_mask, masked_seq_positions, masked_seq_ids,
        #  masked_dot_positions, masked_dot_ids] = [
        #     np.array(each) for each in [
        #         input_seq_ids, input_seq_mask, input_dot_ids, input_dot_mask, masked_seq_positions, masked_seq_ids,
        #         masked_dot_positions, masked_dot_ids
        #     ]
        # ]
        [input_seq_ids, input_dot_ids, input_mask, masked_seq_positions, masked_seq_ids,
         masked_dot_positions, masked_dot_ids] = [
            np.array(each) for each in [
                input_seq_ids, input_dot_ids, input_mask, masked_seq_positions, masked_seq_ids,
                masked_dot_positions, masked_dot_ids
            ]
        ]

        masked_seq_labels = np.ones(input_seq_ids.shape, dtype='int64') * -1
        # store number of masked tokens in index
        for i, (masked_seq_position, masked_seq_label) in enumerate(zip(masked_seq_positions, masked_seq_labels)):
            index = self.max_pred_length

            first_pad = np.nonzero(masked_seq_position == -1)[0]

            if len(first_pad) != 0:
                index = first_pad[0]  # update `index` to the first padding

            # None padding masked_lm_ids
            # print(masked_seq_label)
            # print(index)
            # print(max_seq_length)
            # print(masked_seq_position[:index])

            masked_seq_label[masked_seq_position[:index]] = masked_seq_ids[i, :index]

        masked_dot_labels = np.ones(input_dot_ids.shape, dtype='int64') * -1
        for i, (masked_dot_position, masked_dot_label) in enumerate(zip(masked_dot_positions, masked_dot_labels)):
            index = self.max_pred_length
            # store number of masked tokens in index
            first_pad = np.nonzero(masked_dot_position == -1)[0]
            if len(first_pad) != 0:
                index = first_pad[0]  # update `index` to the first padding
                # index = padded_mask_indices.numpy()[0]  # update `index` to the first padding
            masked_dot_label[masked_dot_position[:index]] = masked_dot_ids[i, :index]
        # return [input_seq_ids, input_dot_ids, input_seq_mask, input_dot_mask, masked_seq_labels, masked_dot_labels]
        return [input_seq_ids, input_dot_ids, input_mask, masked_seq_labels, masked_dot_labels]

    def unpack_field(self, examples, field):  # get batch examples (dictionary) values by key
        return [e[field] for e in examples]


class NormCollator:
    def __init__(self, data_type):
        self.data_type = data_type
        assert isinstance(data_type, str) and data_type in ['train', 'val', 'test']

    def __call__(self, examples):

        seqs = self.unpack_field(examples, 'seq')
        dots = self.unpack_field(examples, 'dot')
        max_seq_len = max([len(each) for each in seqs])

        seqs_padding = np.zeros((len(seqs), max_seq_len), dtype=np.int32)
        dots_padding = np.zeros((len(dots), 4, max_seq_len), dtype=np.int32)

        for i, seq in enumerate(seqs):
            seqs_padding[i, :len(seq)] = seq
        for i, dot in enumerate(dots):
            dots_padding[i, :, :len(dot[0])] = np.array(dot, dtype=np.int32)

        seqs = paddle.to_tensor(seqs_padding, dtype='int32')
        dots = paddle.to_tensor(dots_padding, dtype='int32')

        if self.data_type != 'test':
            labels = self.unpack_field(examples, 'label')
            labels_padding = np.ones((len(labels), max_seq_len), dtype=np.float) * -1
            for i, label in enumerate(labels):
                labels_padding[i, :len(label)] = label
            labels = paddle.to_tensor(labels_padding, dtype='float32')
            # assert (seqs.shape == dots.shape == labels.shape)
            return seqs, dots, labels

        else:
            # assert (seqs.shape == dots.shape)
            return seqs, dots

    def unpack_field(self, examples, field):  # get batch examples (dictionary) values by key
        return [e[field] for e in examples]


class GraphCollator:
    def __init__(self, data_type):
        self.data_type = data_type
        assert isinstance(data_type, str) and data_type in ['train', 'val', 'test']

    def __call__(self, examples):
        graph_list = self.unpack_field(examples, 'graph')
        graph_batch = pgl.Graph.disjoint(graph_list)
        graph_batch.tensor()
        if self.data_type != 'test':
            label_list = self.unpack_field(examples, 'label')
            label_batch = paddle.to_tensor(np.concatenate(label_list))
            return {'graph': graph_batch.tensor(), 'label': label_batch}

        else:
            return {'graph': graph_batch}

    def unpack_field(self, examples, field):  # get batch examples (dictionary) values by key
        return [e[field] for e in examples]


if __name__ == '__main__':
    from model_aug import MyTransformer, PretrainLoss
    from create_pretraining_rna import seq2id, dot2id
    from paddle.io import DataLoader

    dataset = PretrainingDataset(input_file='data/pretrain_data/test.pkl', max_pred_length=100)
    collator = PretrainNormCollator(100)
    dataloader = DataLoader(dataset=dataset, batch_size=8, collate_fn=collator, shuffle=False)
    [input_seq_ids, input_dot_ids, input_mask, masked_seq_labels, masked_dot_labels] = dataloader.__iter__().__next__()
    model = MyTransformer(8, 8, 512, 512 * 3, 6, 5, 0.2)
    seq_pred, dot_pred = model.pretrain_forward(tokens=input_seq_ids, structures=input_dot_ids,
                                                attention_mask=input_mask)
    loss = PretrainLoss(len(seq2id), len(dot2id))
    output = loss(seq_pred, dot_pred, masked_seq_labels, masked_dot_labels)
    import pandas as pd

    df = pd.read_csv('/code/data/rna/RNA_1500_valid_seq_dot.csv')
    df = df[:10]
    input_seq_ids[0]
