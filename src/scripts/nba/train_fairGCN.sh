python train_fairGNN.py \
        --seed=42 \
        --epochs=2000 \
        --model=GCN \
        --dataset=nba \
        --num-hidden=128 \
        --acc=0.70 \
        --roc=0.76 \
        --alpha=10 \
        --beta=1
