# AI-Studio-螺旋桨RNA结构预测竞赛第5名方案

## 项目描述
螺旋桨RNA结构预测竞赛第5名方案, 主要采用transformer + BiGRU的组合模型，最后对不同参数的模型在不同checkpoint下的推理结果进行集成. 

**所有程序代码以及notebook文件都存放在work目录下**

具体的模型构建思路及调优过程，详细请看： `work/1935513.ipynb`

## 项目结构
```
│  LICENSE
│  README.md
│  requirements.txt  
└─work
    │  1935513.ipynb  ## AI Studio 项目的notebook说明(包括模型等)以及运行命令
    │  const.py
    │  dataset.py
    │  dataset_aug.py
    │  main.py     ## 训练文件
    │  model.py    ## 模型文件
    │  model_aug.py
    │  run.sh
    │  run_graph.sh
    │  run_transformer.sh ## 训练脚本
    │  test.py     ## 推理文件
    │  utils.py
    │  
    ├─inference_checkpoint ## 保存checkpoints
    │  └─transformers_gru
    │            
    ├─log   ## 保存训练日志
    │      log_21051310.txt
    │      
    ├─mydata  ## 数据存放
    │  │  B_board_112_seqs.txt
    │  │  dev.txt
    │  │  seq_cb_ns_256_human_1gram
    │  │  seq_cb_ns_256_human_3gram
    │  │  seq_cb_ns_256_human_5gram
    │  │  test_nolabel.txt
    │  │  train.txt
    │  │  vocab.txt
    │  │  
    │  └─ready_data
    └─vocab  ## 字典
            dot_vocab
            seq_vocab
```
## 使用方式
在AI Studio上通过`1935513.ipynb`或终端 [运行本项目]()

**安装依赖**

- 在终端或notebook页面运行 `!pip install -r requirements.txt` 安装运行环境中的依赖包

**不训练，复现榜单上第五名成绩的代码：**

- 在终端或notebook页面运行脚本 `!python test.py --model_dir inference_checkpoint/transformers_gru --ensemble`

- 结果保存的路径为 `result/ensemble_transformers_gru/ensemble/predict.files.zip`

**训练并推理：**

- 在终端或notebook页面运行脚本 `!bash run_transformer.sh` 启动不同参数组合的模型**训练**并保存相应的checkpoints.
- 在终端或notebook页面运行 `!python test.py --model_dir inference_checkpoint/transformers_gru --ensemble` 进行推理结果保存路径`result/ensemble_transformers_gru/ensemble/predict.files.zip`