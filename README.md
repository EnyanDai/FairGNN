# FairGNN 

A PyTorch implementation of "Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information" (WSDM 2021). [[paper]](https://arxiv.org/pdf/2009.01454.pdf)


<div align=center><img src="https://github.com/EnyanDai/FariGNN/blob/main/framework.png" width="700"/></div>

## Requirements

```
torch==1.2.0
DGL=0.4.3
```

## Run the code
After installation, you can clone this repository
```
git clone https://github.com/EnyanDai/FariGNN.git
cd FairGNN/src
python train_fairGNN.py \
        --seed=42 \
        --epochs=2000 \
        --model=GCN \
        --sens_number=200 \
        --dataset=pokec_z \
        --num-hidden=128 \
        --acc=0.69 \
        --roc=0.76 \
        --alpha=100 \
        --beta=1
```
## Model Selection
During the training phase, we will select the best epoch based on the performance on the validation set. More speciafically, the selection rules are: 

1. We only care about the epochs that the accuracy and roc socre of the FairGNN on the validation set are higher than the thresholds (defined by --acc and --roc).
2. We will select the epoch whose summation of parity and equal opportunity is the smallest.

## Data Set
1. Pokec_z and Pokec_n are stored in [`dataset\pokec`](https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec) as `region_job.xxx` and `region_job_2.xxx`, respectively.
They are sampled from [soc_Pokec](http://snap.stanford.edu/data/soc-Pokec.html). 

```
@inproceedings{takac2012data,
  title={Data analysis in public social networks},
  author={Takac, Lubos and Zabovsky, Michal},
  booktitle={International scientific conference and international workshop present day trends of innovations},
  volume={1},
  number={6},
  year={2012}
```
2. NBA is stored in [`dataset\NBA`](https://github.com/EnyanDai/FairGNN/tree/main/dataset/NBA) as `nba.xxx`
It is collected with through the Twitter social network and the players' information on [Kaggle](https://www.kaggle.com/noahgift/social-power-nba)
## Reproduce the results

***Please use DGL 0.4.3***, the version of the DGL can affect the results a lot.
All the hyper-parameters settings are included in [`src\scripts`](https://github.com/EnyanDai/FariGNN/tree/main/src/scripts) folder.

To reproduce the performance reported in the paper, you can run the bash files in folder `src\scripts`.
```
bash scripts/pokec_z/train_fairGCN.sh
```
Here are some example results:
<div align=center><img src="https://github.com/EnyanDai/FariGNN/blob/main/result.png" width="500"/></div>

## PyG verision of FairGNN
Thanks to the great work of the [PyG-Debias](https://github.com/yushundong/PyGDebias) contributors, the PyG verision of FairGNN is available now. Please check the [PyG-Debias package](https://github.com/yushundong/PyGDebias).



## Cite

If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{dai2021say,
  title={Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information},
  author={Dai, Enyan and Wang, Suhang},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  pages={680--688},
  year={2021}
}
```
