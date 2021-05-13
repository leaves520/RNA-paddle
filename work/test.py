import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import shutil
import paddle
import pgl
import pgl.utils.data.dataloader as gd
import zipfile
from paddle.io import Dataset, DataLoader
from model import MyTransformer, GraphTransformer
from dataset import load_test_data, MyDataset, GraphCollator, NormCollator, process_vocabulary
import argparse

def inference(name, n_gram, model_path, config_path):
    seq_vocab, bracket_vocab = process_vocabulary(n_gram=n_gram)
    test = load_test_data(name=name)
    test_data = MyDataset(data_label=test, data_type='test_b', seq_vocab=seq_vocab,
                          bracket_vocab=bracket_vocab, n_gram=n_gram, process=False)

    collate_test = NormCollator(data_type='test')
    dataloader_test = DataLoader(test_data, batch_size=4,
                                 shuffle=False, drop_last=False, collate_fn=collate_test)

    config = pickle.load(open(config_path, 'rb'))
    model = MyTransformer(**config)
    model.load_dict(paddle.load(model_path))
    model.eval()
    with paddle.no_grad():
        result = []
        for index, (seq, dot) in enumerate(dataloader_test):
            mask_pos = (seq != 0)
            attention_mask = paddle.matmul(
                mask_pos.astype('float32').unsqueeze(-1).unsqueeze(1),
                mask_pos.astype('float32').unsqueeze(-1).unsqueeze(1),
                transpose_y=True
            )
            inf = paddle.full(attention_mask.shape, -1e9)
            attention_mask = paddle.where(attention_mask == 0, inf, paddle.zeros(attention_mask.shape))
            out = model(seq, dot, attention_mask).squeeze(-1).numpy()
            for i in range(len(mask_pos)):
                stop = np.nonzero((seq == 0).numpy()[i])[0]
                if len(stop) == 0:
                    result.append(out[i])
                else:
                    result.append(out[i, :stop[0].item()])
    return result


def graph_inference(n_gram, model_path):
    seq_vocab, bracket_vocab = process_vocabulary(n_gram=n_gram)
    test = load_test_data()
    test_data = MyDataset(data_label=test, data_type='test', seq_vocab=seq_vocab,
                          bracket_vocab=bracket_vocab, n_gram=n_gram, process=False, graph=True)

    collate_test = GraphCollator(data_type='test')
    dataloader_test = gd.Dataloader(test_data, batch_size=8,
                                    shuffle=False, drop_last=False, collate_fn=collate_test)

    config = pickle.load(open(os.path.join(model_path, 'model_config'), 'rb'))
    model = GraphTransformer(**config)
    model.load_dict(paddle.load(os.path.join(model_path, 'best_model.pt')))
    model.eval()
    with paddle.no_grad():
        result = []
        for index, batch in enumerate(dataloader_test):
            graph_batch = batch['graph']
            pred_batch = model(graph_batch).squeeze(1).numpy()
            batch_idx = graph_batch.graph_node_id.numpy()
            for i in range(np.max(batch_idx) + 1):
                indices = batch_idx == i
                result.append(pred_batch[indices])

    return result


def write(path, result):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(result)):
        predict = result[i].tolist()
        file_path = os.path.join(path, f"{i + 1}.predict.txt")
        with open(file_path, 'w') as f:
            for each in predict:
                f.write(f"{each}\n")


def zip_file(src_dir):
    zip_name = src_dir + '.zip'
    z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
    # os.removedirs(src_dir)
    z.close()


def ensemble(path):
    files = [os.path.join(path, name, 'predict.files') for name in os.listdir(path)
             if os.path.isdir(os.path.join(path, name))]
    ens_res = []
    for i in range(112):
        file_result = []
        for file in files:
            with open(os.path.join(file, f'{i + 1}.predict.txt')) as f:
                data = [float(item.strip()) for item in f.readlines()]
                file_result.append(data)

        ens_res.append(np.mean(file_result, axis=0))
    return ens_res


def ensemble_1(path):
    files = [os.path.join(path, name, 'predict.files') for name in os.listdir(path)
             if os.path.isdir(os.path.join(path, name))]

    ens_res = []
    for i in range(444):
        file_result = []
        for file in files:
            with open(os.path.join(file, f'{i + 1}.predict.txt')) as f:
                data = [float(item.strip()) for item in f.readlines()]
                file_result.append(data)

        cur_mean = np.mean(file_result, axis=0)
        cur_max = np.max(file_result, axis=0)
        cur_min = np.min(file_result, axis=0)
        res = np.where(cur_mean >= 0.9, cur_max, cur_mean)
        res = np.where(cur_mean <= 0.1, cur_min, res)
        ens_res.append(res)
    return ens_res


def arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--model_dir", type=str, required=True, help="single model dir or ensembles model dir")
    args.add_argument("--n_gram", type=int, default=3)
    args.add_argument("--ensemble", default=False, action="store_true")
    return args.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    board = 'b'

    if not args.ensemble:
        model_name = [each for each in os.listdir(args.model_dir) if each.endswith('.pt')][0]
        model_path = os.path.join(args.model_dir, model_name)
        config_path = os.path.join(args.model_dir, 'model_config')
        result = inference(name=board, n_gram=args.n_gram, model_path=model_path, config_path=config_path)
        save_dir_name = '_'.join(model_path.split('/')[-2:])[:-3]
        save_path = f"result/{save_dir_name}_{board}/predict.files"
        write(save_path, result)
        zip_file(save_path)
        shutil.rmtree(save_path)
    else:

        model_names = []
        config_names = []
        for paths, dirnames, filenames in os.walk(args.model_dir):
            for file in filenames:
                dir_name = paths.split('/')[-1]
                if file.endswith('.pt'):
                    model_names.append(os.path.join(paths, file))
                    config_names.append(os.path.join(paths, 'model_config'))

        first_save_dir = args.model_dir.split('/')[-1]
        for model_path, config_path in zip(model_names, config_names):
            n_gram = int(model_path.split('/')[-2][0])
            result = inference(name=board, n_gram=n_gram, model_path=model_path, config_path=config_path)
            save_dir_name = '_'.join(model_path.split('/')[-2:])[:-3]
            save_path = f"result/ensemble_{first_save_dir}/{save_dir_name}_{board}/predict.files"
            write(save_path, result)
            zip_file(save_path)
        
        path = f'result/ensemble_{first_save_dir}/'
        res = ensemble(path)
        save_path = f'result/ensemble_{first_save_dir}/ensemble/predict.files'
        write(save_path, res)
        zip_file(save_path)
        shutil.rmtree(save_path)       
    # board = 'b'
    # model_path = "checkpoints/transformer_aug/test/best_model_0.07056_0.23625.pt"
    # config_path = "checkpoints/transformer_aug/test/model_config"
    # result = inference(name=board, n_gram=3, model_path=model_path, config_path=config_path)
    # save_path = f"result/transformer_aug/{model_path.split('/')[3][:-3]}_{board}/predict.files"
    # write(save_path, result)
    # zip_file(save_path)
    # shutil.rmtree(save_path)

    # model_path = "/code/RNA_lzl/checkpoints/graph_256_skip_gate_4_4_0.267/"
    # result = graph_inference(n_gram=1, model_path=model_path)
    # save_path = f"result/{model_path.split('/')[4]}/predict.files"
    # write(save_path, result)
    # zip_file(save_path)
    # shutil.rmtree(save_path)

    # path = 'result/ensemble_1_3/'
    # res = ensemble(path)
    # save_path = 'result/ensemble_1_3/ensemble/predict.files'
    # write(save_path, res)
    # zip_file(save_path)
    # shutil.rmtree(save_path)

    # res = ensemble_1("/code/RNA_lzl/result/ensemble_transformer_graph/")
    # # res1 = ensemble("/code/RNA_lzl/result/ensemble_transformer_graph/")
    # save_path = 'result/transformer+graph/predict.files'
    # write(save_path, res)
    # zip_file(save_path)
    # shutil.rmtree(save_path)
