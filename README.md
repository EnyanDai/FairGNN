# FairGNN 

A PyTorch implementation of "FairGNN: Eliminating the Discrimination in Graph Neural
Networks with Limited Sensitive Attribute Information" (WSDM 2021). [[paper]](https://arxiv.org/pdf/2009.01454.pdf)


<div align=center><img src="https://github.com/EnyanDai/FariGNN/blob/main/framework.png" width="700"/></div>

## Abstract 
Graph neural networks (GNNs) have shown great power in modeling graph structured data. However, similar to other machine learning models, GNNs may make predictions biased on protected sensitive attributes, e.g., skin color, gender, and nationality. 
Because machine learning algorithms including GNNs are trained to faithfully reflect the distribution of the training data which often contains historical bias towards sensitive attributes. In addition, the discrimination in GNNs can be magnified by graph structures and the message-passing mechanism. As a result, the applications of GNNs in sensitive domains such as crime rate prediction would be largely limited. Though extensive studies of fair classification have been conducted on i.i.d data, methods to address the problem of discrimination on non-i.i.d data are rather limited. Furthermore, 
the practical scenario of sparse annotations in sensitive attributes is rarely considered in existing works. Therefore, we study the novel and important problem of learning fair GNNs with limited sensitive attribute information. FairGNN is proposed to eliminate the bias of GNNs whilst maintaining high node classification accuracy by leveraging graph structures and limited sensitive information. Our theoretical analysis shows that FairGNN can ensure the fairness of GNNs under mild conditions given limited nodes with known sensitive attributes. Extensive experiments on real-world datasets also demonstrate the effectiveness of FairGNN in debiasing and keeping high accuracy.

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
Pokec_z and Pokec_n are sampled from [soc_Pokec](http://snap.stanford.edu/data/soc-Pokec.html).

```
@inproceedings{takac2012data,
  title={Data analysis in public social networks},
  author={Takac, Lubos and Zabovsky, Michal},
  booktitle={International scientific conference and international workshop present day trends of innovations},
  volume={1},
  number={6},
  year={2012}
```
NBA is collected with through the Twitter social network and the players' information on [Kaggle](https://www.kaggle.com/noahgift/social-power-nba)
## Reproduce the results
All the hyper-parameters settings are included in [`src\scripts`](https://github.com/EnyanDai/FariGNN/tree/main/src/scripts) folder.

To reproduce the performance reported in the paper, you can run the bash files in folder `src\scripts`.
```
bash scripts/pokec_z/train_fairGCN.sh
```






## Cite

If you find this repo to be useful, please cite our paper. Thank you.
```
@article{dai2020fairgnn,
  title={FairGNN: Eliminating the Discrimination in Graph Neural Networks with Limited Sensitive Attribute Information},
  author={Dai, Enyan and Wang, Suhang},
  journal={arXiv preprint arXiv:2009.01454},
  year={2020}
}
```