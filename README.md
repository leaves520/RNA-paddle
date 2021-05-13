# AI-Studio-螺旋桨RNA结构预测竞赛第5名方案

## 项目描述
螺旋桨RNA结构预测竞赛第5名方案, 主要采用transformer + BiGRU的组合模型，最后对不同参数的模型在不同checkpoint下的推理结果进行集成. 

具体的模型构建思路及调优过程看 -xxx.ipynb

## 项目结构
```
-README.MD
-xxx.ipynb
```
## 使用方式
在AI Studio上[运行本项目]()

1.在终端或notebook页面运行 `!pip install -r requirements.txt` 安装运行环境中的依赖包

2.在终端或notebook页面运行脚本 `!bash run_transformer.sh` 启动不同参数组合的模型**训练**并保存相应的checkpoints.

3.在终端或notebook页面运行 `!python test.py --model_dir inference_checkpoint/transformers_gru --ensemble` 进行推理
