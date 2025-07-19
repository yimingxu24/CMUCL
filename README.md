# Text-Attributed Graph Anomaly Detection via Multi-Scale Cross- and Uni-Modal Contrastive Learning

<img src="https://github.com/yimingxu24/CMUCL/blob/main/pipeline.svg" width="60%">

## 0. Python environment setup with Conda

```
torch==1.12.1
torch-cluster==1.6.0
torch-geometric==2.1.0.post1
torch-scatter==2.0.9
torch-sparse==0.6.15
torch-spline-conv==1.2.1
torchaudio==0.12.1
torchvision==0.13.1
transformers==4.24.0
Python==3.8.13
```

## 1. TAG datasets

Download the datasets [here](https://drive.google.com/drive/folders/1Suws6A-v0jBQpKeMphD0CEgxBjK-72AC?usp=sharing), unzip and move it to `./data`

| Dataset     | Nodes   | Edges    | Avg. doc length | Attributes | Anomalies |
|-------------|---------|----------|-----------------|------------|-----------|
| Citeseer    | 3,186   | 3,432    | 153.94          | 768        | 128       |
| Pubmed      | 19,717  | 90,368   | 256.08          | 768        | 788       |
| History     | 41,551  | 369,252  | 228.36          | 768        | 1,662     |
| Photo       | 48,362  | 512,933  | 150.25          | 768        | 1,934     |
| Computers   | 87,229  | 742,792  | 93.16           | 768        | 3,490     |
| Children    | 76,875  | 1,574,664| 209.12          | 768        | 3,076     |
| ogbn-Arxiv  | 169,343 | 1,210,112| 179.70          | 768        | 6,774     |
| CitationV8  | 1,106,759| 6,396,265| 148.77         | 768        | 44,270    |

## 2. Training and inference:
```
python main_train.py --dataset Citeseer --lr 0.0002 --epoch_num 2 --gamma 0.005

python main_train.py --dataset Pubmed --lr 2e-05 --epoch_num 2 --gamma 0.001

python main_train.py --dataset History --lr 2e-05 --epoch_num 2 --gamma 0.5

python main_train.py --dataset Photo --lr 5e-05 --epoch_num 3 --gamma 0.001

python main_train.py --dataset Computers --lr 2e-05 --epoch_num 3 --gamma 0.01

python main_train.py --dataset Children --lr 5e-05 --epoch_num 2 --gamma 0.5

python main_train.py --dataset Arxiv --lr 1e-05 --epoch_num 2 --gamma 0.01

python main_train.py --dataset CitationV8 --lr 2e-5 --epoch_num 2 --gamma 0.5
```
