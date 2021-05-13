import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import pgl
import pickle
import os


class MyTransformer(nn.Layer):
    def __init__(
            self,
            num_layers,
            n_heads,
            d_model,
            hidden_inter,
            word_vocab_size,
            struct_vocab_size,
            dropout,
            token_weights=None,
            structure_weights=None,
            output_logit=False
    ):
        super(MyTransformer, self).__init__()

        self.config = {k: v for k, v in locals().items() if k not in ['self', '__class__',
                                                                      'token_weights', 'structure_weights']}
        self.token_embedding = nn.Embedding(word_vocab_size, d_model)
        if token_weights is not None:
            self.token_embedding.weight.set_value(token_weights)
            for param in self.token_embedding.parameters():
                param.stop_gradient = True

        self.structure_embedding = nn.Embedding(struct_vocab_size, d_model)
        if structure_weights is not None:
            self.structure_embedding.weight.set_value(structure_weights)
            # for param in self.structure_embedding.parameters():
            #     param.stop_gradient = True

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_inter,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )
        self.output_logit = output_logit
        self.fc = nn.Linear(2 * d_model, d_model)

        # self.linear = nn.Linear(2 * d_model, 1)
        self.activation = nn.Sigmoid()

        self.seq_linear = nn.Linear(d_model, word_vocab_size)
        self.dot_linear = nn.Linear(d_model, struct_vocab_size)
        self.dropout = nn.Dropout(dropout)
        # self.bi_gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, dropout=0.1,
        #                      direction='bidirect')
        self.bi_gru = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=2, dropout=0.1,
                             direction='bidirect')
        self.linear_project = nn.Sequential(nn.Linear(2 * d_model, d_model // 2),
                                            nn.ReLU(),
                                            nn.Linear(d_model // 2, 1))

    def forward(self, tokens, structures, attention_mask):
        struc_emb = self.structure_embedding(structures)
        struc_emb = struc_emb[:, 0] * 0.4 + struc_emb[:, 1] * 0.8 + struc_emb[:, 2] * 1.2 + struc_emb[:, 3] * 1.6
        # struc_emb = paddle.mean(self.structure_embedding(structures), axis=1)
        embeddings = paddle.concat([struc_emb, self.token_embedding(tokens)], axis=-1)
        embeddings = self.dropout(embeddings)
        embeddings = self.fc(embeddings)
        embeddings = self.dropout(embeddings)
        output = self.encoder(embeddings, attention_mask)
        output = self.dropout(output)
        output, h = self.bi_gru(output)
        if self.output_logit:
            return output
        else:
            output = self.linear_project(output)
            prob = self.activation(output)
            return prob

    def pretrain_forward(self, tokens, structures, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).astype('float32')
        inf = paddle.full(attention_mask.shape, -1e9)
        attention_mask = paddle.where(attention_mask == 0, inf, paddle.zeros(attention_mask.shape))

        embeddings = paddle.concat([self.structure_embedding(structures), self.token_embedding(tokens)], axis=-1)
        embeddings = self.dropout(embeddings)
        embeddings = self.fc(embeddings)
        embeddings = self.dropout(embeddings)
        output = self.encoder(embeddings, attention_mask)
        output = self.dropout(output)
        seq_pred = self.seq_linear(output)
        dot_pred = self.dot_linear(output)
        return seq_pred, dot_pred

    def get_embeddings(self, tokens, structures, attention_mask):
        embeddings = self.structure_embedding(structures) + self.token_embedding(tokens)
        output = self.encoder(embeddings, attention_mask)
        return output

    @classmethod
    def from_pretrained(cls, model_path):
        config = pickle.load(open(os.path.join(model_path, 'model_config'), 'rb'))
        model = cls(**config)
        model.load_dict(paddle.load(os.path.join(model_path, 'best_model.pt')))
        return model


class PretrainLoss(nn.Layer):
    def __init__(self, seq_vocab_size, dot_vocab_size):
        super(PretrainLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.seq_vocab_size = seq_vocab_size
        self.dot_vocab_size = dot_vocab_size

    def forward(self, seq_pred, dot_pred, seq_labels, dot_labels):
        seq_loss = self.loss_fn(seq_pred.reshape(shape=(-1, self.seq_vocab_size)), seq_labels.flatten())
        dot_loss = self.loss_fn(dot_pred.reshape(shape=(-1, self.dot_vocab_size)), dot_labels.flatten())
        return seq_loss + dot_loss


class TextCNN(nn.Layer):
    def __init__(self, seq_len, dot_len, max_seqlen, hidden_dim=128, num_tasks=2):
        super(TextCNN, self).__init__()
        self.num_tasks = num_tasks
        self.max_seqlen = max_seqlen

        self.embed_seq = nn.Embedding(seq_len, hidden_dim)
        self.embed_dot = nn.Embedding(dot_len, hidden_dim)

        self.conv1 = nn.Conv2D(in_channels=2, out_channels=3, kernel_size=(2, hidden_dim), stride=1,
                               padding=0)  # input: batch,chanals,seq_length,dim
        self.conv2 = nn.Conv2D(in_channels=2, out_channels=3, kernel_size=(3, hidden_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2D(in_channels=2, out_channels=3, kernel_size=(4, hidden_dim), stride=1, padding=0)

        self.linear1 = nn.Linear((max_seqlen - 2) + 1, max_seqlen)
        self.linear2 = nn.Linear((max_seqlen - 3) + 1, max_seqlen)
        self.linear3 = nn.Linear((max_seqlen - 4) + 1, max_seqlen)

        self.out = nn.Linear(3, 2)
        self.activate = nn.Softmax(axis=-1)

    def forward(self, seq_x, dot_x):  # batch,seq_len
        seq_x = self.embed_seq(seq_x).unsqueeze(1)  # batch,seq_len,dim
        dot_x = self.embed_dot(dot_x).unsqueeze(1)  # batch,seq_len,dim
        seq_dot_x = paddle.concat([seq_x, dot_x], axis=1)

        x1 = self.conv1(seq_dot_x)  # batch,out_channals,1,1
        x2 = self.conv2(seq_dot_x)
        x3 = self.conv3(seq_dot_x)

        x1 = paddle.mean(x1, axis=1).squeeze(2)
        x2 = paddle.mean(x2, axis=1).squeeze(2)
        x3 = paddle.mean(x3, axis=1).squeeze(2)

        x1 = self.linear1(x1).unsqueeze(-1)
        x2 = self.linear2(x2).unsqueeze(-1)
        x3 = self.linear3(x3).unsqueeze(-1)

        x_cat = paddle.concat([x1, x2, x3], axis=-1)
        y = self.out(x_cat)

        return self.activate(y)[:, :, 0]


class TextCNN_BiGRU(nn.Layer):
    def __init__(self, seq_len, dot_len, n_gram=3, filter_nums=3, hidden_dim=128, num_tasks=2):
        super(TextCNN_BiGRU, self).__init__()
        self.n_gram = n_gram
        self.num_tasks = num_tasks

        self.embed_seq = nn.Embedding(seq_len, hidden_dim)
        self.embed_dot = nn.Embedding(dot_len, hidden_dim)

        # input: batch,chanals,seq_length,dim
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels=2, out_channels=filter_nums, kernel_size=(2, hidden_dim), stride=1,
                      padding=[1, 0, 0, 0]), nn.BatchNorm2D(filter_nums), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2D(in_channels=2, out_channels=filter_nums, kernel_size=(3, hidden_dim), stride=1, padding=[1, 0]),
            nn.BatchNorm2D(filter_nums), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2D(in_channels=2, out_channels=filter_nums, kernel_size=(4, hidden_dim), stride=1,
                      padding=[2, 1, 0, 0]), nn.BatchNorm2D(filter_nums), nn.ReLU())

        self.bi_gru = nn.GRU(input_size=3 * filter_nums, hidden_size=3 * filter_nums, num_layers=2, dropout=0.1,
                             direction='bidirect')
        self.linear_project = nn.Sequential(nn.Linear(2 * 3 * filter_nums, 6), nn.Dropout(0.1), nn.Linear(6, 2))
        self.activate = nn.Softmax(axis=-1)

    def forward(self, seq_x, dot_x):  # batch,seq_len
        seq_x = self.embed_seq(seq_x).unsqueeze(1)  # batch,seq_len,dim -> batch,1,seq_len,dim
        dot_x = self.embed_dot(dot_x).unsqueeze(1)
        seq_dot_x = paddle.concat([seq_x, dot_x], axis=1)

        x1 = paddle.transpose(self.conv1(seq_dot_x), [0, 3, 2, 1])  # batch,1,seq_len,out_channals
        x2 = paddle.transpose(self.conv2(seq_dot_x), [0, 3, 2, 1])
        x3 = paddle.transpose(self.conv3(seq_dot_x), [0, 3, 2, 1])

        x_cat = paddle.concat([x1, x2, x3], axis=-1).squeeze(1)
        out, h = self.bi_gru(x_cat)
        y = self.linear_project(out)
        return self.activate(y)[:, :, 0]


class GraphTransformer(nn.Layer):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 num_classes=1,
                 num_layers=6,
                 num_heads=4,
                 feat_drop=0.3,
                 attn_drop=0.3,
                 concat=True,
                 skip_feat=True,
                 gate=False,
                 layer_norm=True,
                 activation='relu',
                 layer_dropout=0.3
                 ):
        super(GraphTransformer, self).__init__()

        self.config = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat
        self.skip_feat = skip_feat
        self.gate = gate
        self.layer_norm = layer_norm
        self.activation = activation
        self.num_classes = num_classes
        self.encoder_list = nn.LayerList()
        self.edge_embedding = nn.Embedding(5, self.embed_size)
        self.seq_embedding = nn.Embedding(4, self.embed_size)
        self.dot_embedding = nn.Embedding(3, self.embed_size)
        self.layer_dropout = layer_dropout
        self.output_linear = nn.Linear(self.hidden_size * self.num_heads, num_classes)
        for i in range(num_layers):
            if i == 0:
                self.encoder_list.append(
                    pgl.nn.TransformerConv(
                        input_size=self.embed_size,
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        feat_drop=self.feat_drop,
                        attn_drop=self.attn_drop,
                        concat=self.concat,
                        skip_feat=self.skip_feat,
                        gate=self.gate,
                        layer_norm=self.layer_norm
                    )
                )
            else:
                self.encoder_list.append(
                    pgl.nn.TransformerConv(
                        input_size=self.hidden_size * self.num_heads,
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        feat_drop=self.feat_drop,
                        attn_drop=self.attn_drop,
                        concat=self.concat,
                        skip_feat=self.skip_feat,
                        gate=self.gate,
                        layer_norm=self.layer_norm
                    )
                )
            self.encoder_list.append(nn.Dropout(self.layer_dropout))

    def forward(self, graph):
        seq_feat = self.seq_embedding(graph.node_feat['one_hot_seq_feature'].squeeze(1))
        dot_feat = self.dot_embedding(graph.node_feat['one_hot_dot_feature'].squeeze(1))
        node_feat = seq_feat + dot_feat
        # node_feat = seq_feat
        edge_feat = self.edge_embedding(graph.edge_feat['one_hot_feature'].squeeze(1))

        for layer in self.encoder_list:
            if isinstance(layer, nn.Dropout):
                node_feat = layer(node_feat)
            else:
                node_feat = layer(graph, node_feat, None)

        node_feat = self.output_linear(node_feat)
        return F.sigmoid(node_feat)

    def get_embeddings(self, graph):
        seq_feat = self.seq_embedding(graph.node_feat['one_hot_seq_feature'].squeeze(1))
        dot_feat = self.dot_embedding(graph.node_feat['one_hot_dot_feature'].squeeze(1))
        node_feat = seq_feat + dot_feat
        # node_feat = seq_feat
        edge_feat = self.edge_embedding(graph.edge_feat['one_hot_feature'].squeeze(1))

        for layer in self.encoder_list:
            if isinstance(layer, nn.Dropout):
                node_feat = layer(node_feat)
            else:
                node_feat = layer(graph, node_feat, None)
        return node_feat


class GAT(nn.Layer):
    """Implement of GAT
    """

    def __init__(
            self,
            embed_size,
            num_class,
            num_layers=1,
            feat_drop=0.6,
            attn_drop=0.6,
            num_heads=8,
            hidden_size=64, ):
        super(GAT, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.gats = nn.LayerList()
        self.embed_size = embed_size
        self.seq_embedding = nn.Embedding(4, self.embed_size)
        self.dot_embedding = nn.Embedding(3, self.embed_size)

        for i in range(self.num_layers):
            if i == 0:
                self.gats.append(
                    pgl.nn.GATConv(
                        self.embed_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation='elu'))
            elif i == (self.num_layers - 1):
                self.gats.append(
                    pgl.nn.GATConv(
                        self.num_heads * self.hidden_size,
                        self.num_class,
                        self.feat_drop,
                        self.attn_drop,
                        1,
                        concat=False,
                        activation=None))
            else:
                self.gats.append(
                    pgl.nn.GATConv(
                        self.num_heads * self.hidden_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation='elu'))

    def forward(self, graph):
        seq_feat = self.seq_embedding(graph.node_feat['one_hot_seq_feature'].squeeze(1))
        dot_feat = self.dot_embedding(graph.node_feat['one_hot_dot_feature'].squeeze(1))
        node_feat = seq_feat + dot_feat
        for m in self.gats:
            node_feat = m(graph, node_feat)
        return F.sigmoid(node_feat)


class MSE(nn.Layer):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = paddle.nn.loss.MSELoss(reduction='none')

    def forward(self, pre, true, mask_pos):
        assert pre.shape == true.shape == mask_pos.shape
        loss = self.mse(pre, true)
        n = np.sum(mask_pos.numpy())
        return paddle.sum(paddle.where(mask_pos, loss, paddle.zeros(loss.shape))) / n


class SoftEntropyLoss(nn.Layer):
    def __init__(self, ignore_index):
        super(SoftEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pre, true):
        mask = true == self.ignore_index
        entropy_matrix = true * paddle.log(pre) + (1 - true) * paddle.log(1 - pre)
        n = paddle.sum(1 - mask.astype('float'))
        entropy_matrix = paddle.where(mask, paddle.zeros(entropy_matrix.shape), entropy_matrix)
        return -1 * paddle.sum(entropy_matrix) / n


if __name__ == '__main__':
    model = MyTransformer(4, 4, 128, 128, 10, 10, 0.2)
    model = paddle.DataParallel(model)
    paddle.save({'model': model.state_dict(),
                 'epoch': 20
                 }, path='checkpoints/test/test.pt')

