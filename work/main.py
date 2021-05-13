import os
import time
import numpy as np
import random
import paddle
import shutil
import argparse
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import pgl.utils.data.dataloader as gd
import pickle

from model import MyTransformer, MSE, GraphTransformer
from dataset import MyDataset, load_train_data, process_vocabulary, NormCollator, GraphCollator
from gensim.models import Word2Vec


def get_embedding_weights(save_path, vocab):
    cbow = Word2Vec.load(save_path)
    word2vec = {word: paddle.to_tensor(cbow.wv[word]).unsqueeze(0).astype('float32') for word in
                cbow.wv.vocab.keys()}
    shape = list(word2vec.values())[0].shape
    emb = {vocab.indices[word]: word2vec.get(word, paddle.zeros(shape).astype('float32'))
           for word in vocab.indices.keys()}
    return paddle.concat(list(emb.values()), axis=0)


def rmsd(pre_proba, true_proba, mask, reduction='mean'):
    assert reduction in ['mean', 'none']
    if mask is not None:
        n = np.sum(mask)
        sq = (pre_proba - true_proba) ** 2
        sq = np.where(mask, sq, np.zeros(sq.shape))
    else:
        n = len(pre_proba)
        sq = (pre_proba - true_proba) ** 2
    return np.sqrt(sq.sum() / n) if reduction == "mean" else sq


def arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=500, help="number of training epoch")
    args.add_argument("--n_gram", type=int, default=3, help="number of gram to process rna sequence")
    args.add_argument("--patience", type=int, default=50, help="early stop patience")
    args.add_argument("--num_layers", type=int, default=12, help="number of encoder layers")
    args.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
    args.add_argument("--d_model", type=int, default=256, help="input and output dim of encoder layer")
    args.add_argument("--hidden_inter", type=int, default=768, help="hidden dim in encoder layer, often 3 * d_model")
    args.add_argument("--dropout_rate", type=float, default=0.1, help="drop out ratio")
    args.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    args.add_argument("--warmup_steps", type=int, default=1000, help="number of steps to warm up learning rate")
    args.add_argument("--weight_decay", type=float, default=1e-5, help="optimizer weight decay")
    args.add_argument("--batch_size", type=int, default=32, help="training batch size")
    args.add_argument("--attention_window_size", type=int, default=None, help="number of attention context window size to set in encoder")
    args.add_argument("--save_dir", type=str, default="inference_checkpoints/test", help="folder to save checkpoint ")
    args.add_argument("--process_data", default=False, action="store_true", help="whether to reprocess data")
    args.add_argument("--seq_emb", type=str, default=None, help="pre-trained sequence embedding path(Must be instance of gensim Word2vec object)")
    args.add_argument("--dot_emb", type=str, default=None, help="pre-trained structure embedding path(Must be instance of gensim Word2vec object)")
    args.add_argument("--seed", type=int, default=1, help="global random seed")
    
    # for Graph Training
    args.add_argument("--graph", default=False, action="store_true", help="whether to use Graph pipline to train, default false")
    args.add_argument("--skip", default=False, action="store_true", help="whether to add skip connection in GraphTransformer")
    args.add_argument("--gate", default=False, action="store_true", help="whether to add gate structure in GraphTransformer")

    return args.parse_args()


def neighbour_mask(i, length, window):
    left = max(0, i - window)
    right = min(length, i + window + 1)
    mask = [-1e9] * length
    for i in range(left, right):
        mask[i] = 0.
    return mask


def normal_train(dataloader_train, model, criterion, optimizer):
    model.train()
    for index, (seq, dot, labels) in enumerate(dataloader_train):
        mask_pos = (seq != 0)
        attention_mask = paddle.matmul(
            mask_pos.astype('float32').unsqueeze(-1).unsqueeze(1),
            mask_pos.astype('float32').unsqueeze(-1).unsqueeze(1),
            transpose_y=True
        )
        inf = paddle.full(attention_mask.shape, -1e9)
        attention_mask = paddle.where(attention_mask == 0, inf, paddle.zeros(attention_mask.shape))
        if args.attention_window_size is not None:
            pos_mask = [neighbour_mask(i, attention_mask.shape[-1], args.attention_window_size)
                        for i in range(attention_mask.shape[-1])]
            pos_mask = paddle.to_tensor(pos_mask)
            attention_mask = paddle.where((attention_mask + pos_mask) == 0,
                                          paddle.zeros(attention_mask.shape), inf)

        out = model(seq, dot, attention_mask).squeeze(-1)
        loss = criterion(out, labels, mask_pos)
        if (index + 1) % 30 == 0:
            print("Train process : epochs:{} step:{} loss:{} rmsd:{}".format(epoch + 1, index + 1,
                                                                             loss.numpy(), None))

        date = "Train process " + str(epoch + 1) + ' ' + str(index + 1) + ' ' + str(loss.numpy()) + '\n'
        log.write(date)
        log.flush()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()


def graph_train(dataloader_train, model, criterion, optimier):
    model.train()
    for index, batch in enumerate(dataloader_train):
        graph_batch = batch['graph']
        label_batch = batch['label']
        pred_batch = model(graph_batch).reshape(label_batch.shape)
        # if any(paddle.isnan(pred_batch).numpy()):
        #     print('nan')
        loss = criterion(pred_batch, label_batch)
        if (index + 1) % 30 == 0:
            # compute rmsd
            sq = (pred_batch - label_batch) ** 2
            batch_id = graph_batch.graph_node_id
            scatter_sq = paddle.scatter(
                x=paddle.zeros(graph_batch.num_graph),
                index=batch_id,
                updates=sq,
                overwrite=False
            )
            num_sq = paddle.scatter(
                x=paddle.zeros(graph_batch.num_graph),
                index=batch_id,
                updates=paddle.ones(graph_batch.num_nodes),
                overwrite=False
            )
            avg_rmsd = paddle.sqrt(scatter_sq / num_sq).mean()
            print("Train process : epochs:{} step:{} loss:{} rmsd:{}".format(epoch + 1, index + 1,
                                                                             loss.numpy(), avg_rmsd.numpy()))

        date = "Train process " + str(epoch + 1) + ' ' + str(index + 1) + ' ' + str(loss.numpy()) + '\n'
        log.write(date)
        log.flush()

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()


def normal_eval(dataloader_val, model, criterion):
    model.eval()
    rmsd_list = []
    loss_list = []
    with paddle.no_grad():
        for index, (seq, dot, labels) in enumerate(dataloader_val):
            mask_pos = (seq != 0)
            attention_mask = paddle.matmul(
                mask_pos.astype('float32').unsqueeze(-1).unsqueeze(1),
                mask_pos.astype('float32').unsqueeze(-1).unsqueeze(1),
                transpose_y=True
            )
            inf = paddle.full(attention_mask.shape, -1e9)
            attention_mask = paddle.where(attention_mask == 0, inf, paddle.zeros(attention_mask.shape))

            out = model(seq, dot, attention_mask).squeeze(-1)
            loss = criterion(out, labels, mask_pos)
            total_rmsd, batch_seq_len = 0, len(seq)

            for index in range(batch_seq_len):
                metric = rmsd(out[index].numpy(), labels[index].numpy(), mask_pos.numpy()[index])
                total_rmsd += metric
            avg_rmsd = total_rmsd / batch_seq_len
            rmsd_list.append(avg_rmsd)
            loss_list.append(loss.numpy())
    return np.mean(rmsd_list), np.mean(loss_list)


def graph_eval(dataloader_val, model, criterion):
    model.eval()
    with paddle.no_grad():
        avg_rmsd = 0
        avg_loss = 0
        for index, batch in enumerate(dataloader_val):
            graph_batch = batch['graph']
            label_batch = batch['label']
            pred_batch = model(graph_batch).reshape(label_batch.shape)
            avg_loss += criterion(pred_batch, label_batch)

            # compute scattered rmsd
            sq = (pred_batch - label_batch) ** 2
            batch_id = graph_batch.graph_node_id
            scatter_sq = paddle.scatter(
                x=paddle.zeros(graph_batch.num_graph),
                index=batch_id,
                updates=sq,
                overwrite=False
            )
            num_sq = paddle.scatter(
                x=paddle.zeros(graph_batch.num_graph),
                index=batch_id,
                updates=paddle.ones(graph_batch.num_nodes),
                overwrite=False
            )
            avg_rmsd += paddle.sqrt(scatter_sq / num_sq).mean()

        avg_rmsd /= len(dataloader_val)
        avg_loss /= len(dataloader_val)
    return float(avg_rmsd.numpy()), float(avg_loss.numpy())


if __name__ == '__main__':
    args = arg_parse()
    print(args)

    assert args.d_model % args.n_heads == 0
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    if not os.path.exists(args.save_dir):
        print(f"Creating save folder: {args.save_dir}")
        os.makedirs(args.save_dir)

    EPOCHS = args.epochs
    train, val = load_train_data()
    seq_vocab, dot_vocab = process_vocabulary(n_gram=args.n_gram)

    seq_emb = get_embedding_weights(args.seq_emb, seq_vocab) if args.seq_emb is not None else None

    dot_emb = get_embedding_weights(args.dot_emb, dot_vocab) if args.dot_emb is not None else None

    train_data = MyDataset(data_label=train, data_type='train', seq_vocab=seq_vocab,
                           bracket_vocab=dot_vocab, n_gram=args.n_gram, process=args.process_data, graph=args.graph)
    val_data = MyDataset(data_label=val, data_type='val', seq_vocab=seq_vocab,
                         bracket_vocab=dot_vocab, n_gram=args.n_gram, process=args.process_data, graph=args.graph)

    Collator = GraphCollator if args.graph else NormCollator
    collate_train = Collator(data_type='train')
    collate_val = Collator(data_type='val')

    DataLoader = gd.Dataloader if args.graph else paddle.io.DataLoader
    dataloader_train = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, drop_last=False, collate_fn=collate_train)
    dataloader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                collate_fn=collate_val)

    if args.graph:

        model = GraphTransformer(
            embed_size=args.d_model,
            hidden_size=args.d_model // args.n_heads,
            num_classes=1,
            num_layers=args.num_layers,
            num_heads=args.n_heads,
            skip_feat=args.skip,
            feat_drop=args.dropout_rate,
            attn_drop=args.dropout_rate,
            gate=args.gate
        )

        print(f"Using Structure: {type(model)}")
    else:
        model = MyTransformer(
            num_layers=args.num_layers, n_heads=args.n_heads, d_model=args.d_model, hidden_inter=args.hidden_inter,
            word_vocab_size=seq_vocab.size, struct_vocab_size=dot_vocab.size, dropout=args.dropout_rate,
            token_weights=seq_emb, structure_weights=dot_emb
        )
    pickle.dump(model.config, open(os.path.join(args.save_dir, 'model_config'), 'wb'))

    criterion = MSE() if not args.graph else paddle.nn.MSELoss(reduction="mean")

    # scheduler = paddle.optimizer.lr.NoamDecay(
    #     d_model=args.d_model, warmup_steps=args.warmup_steps, learning_rate=args.lr
    # )

    optimizer = paddle.optimizer.Adam(
        learning_rate=args.lr, parameters=model.parameters(), weight_decay=args.weight_decay
    )

    log = open('log/log_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log.write('epoch step loss\n')

    print("Training model ...")
    best_rmsd = float('inf')
    best_loss = float('inf')
    patience = args.patience
    p = 0

    model_checkpoint_queue = []
    for epoch in range(EPOCHS):
        if not args.graph:
            normal_train(dataloader_train, model, criterion, optimizer)

            # scheduler.step()
            # print("Current learning rate: {:.5f}".format(scheduler.get_lr()))

            avg_rmsd, avg_loss = normal_eval(dataloader_val, model, criterion)

        else:
            graph_train(dataloader_train, model, criterion, optimizer)
            avg_rmsd, avg_loss = graph_eval(dataloader_val, model, criterion)

        print("\t validation loss: {}, avg_rmsd: {}".format(avg_loss, avg_rmsd))
        date = "\t validation loss " + str(avg_loss) + 'avg_rmsd ' + str(avg_rmsd) + '\n'
        log.write(date)
        log.flush()

        if avg_loss < best_loss:
            p = 0
            print("Current best model is saved .....")
            best_loss = avg_loss
            best_rmsd = avg_rmsd
            cur_model_name = "best_model_{:.5f}_{:.5f}.pt".format(best_loss, best_rmsd)
            paddle.save(model.state_dict(), os.path.join(args.save_dir, cur_model_name))
            model_checkpoint_queue.append(cur_model_name)
            if len(model_checkpoint_queue) > 3:
                model_to_remove = model_checkpoint_queue.pop(0)
                print(f"Remove {model_to_remove}")
                try:
                    os.remove(os.path.join(args.save_dir, model_to_remove))
                except:
                    pass

        else:
            p += 1
            print("Performance in validation not improve.. Current patience is {}/{}".format(p, patience))

        if p >= patience:
            print("early stop!")
            break

        print("\n")

    print("Training finish! The Best rmsd is : {}, Best loss is : {}".format(best_rmsd, best_loss))
    if best_rmsd > 0.265:
        print("Remove!")
        shutil.rmtree(args.save_dir)
