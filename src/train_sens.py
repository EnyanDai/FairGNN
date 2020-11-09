#%%
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy,load_pokec,load_pokec_emb
from models.GCN import GCN
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=41, help='Random seed.')
parser.add_argument('--epochs', type=int, default=800,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

#%%
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data

# dataset = "region_job"
# sens_attr = "region"
# predict_attr = "I_am_working_in_field"
# label_number = 500
# sens_number = 200
# seed = 20
# path="../dataset/pokec/"

dataset = "nba"
sens_attr = "country"
predict_attr = "SALARY"
label_number = 100
sens_number = 50
seed = 20
path = "../dataset/NBA"
test_idx = True
adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                                    seed=seed,test_idx=test_idx)
print(len(idx_test))
#%%
import dgl
from utils import feature_norm
G = dgl.DGLGraph()
G.from_scipy_sparse_matrix(adj)
if dataset=="nba":
    features = feature_norm(features)

#%%
# predict_attr = "relation_to_smoking"
# adj, features, sens, idx_sens_train, idx_val, idx_test,sens = load_pokec(dataset=dataset,predict_attr=predict_attr,number=5000)
sens[sens>0]=1
if sens_attr:
    sens[sens>0]=1
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    # adj = adj.cuda()
    sens = sens.cuda()
    # idx_sens_train = idx_sens_train.cuda()
    # idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()

from sklearn.metrics import accuracy_score,roc_auc_score,f1_score


    


# Train model
t_total = time.time()
best_roc = 0.0
for epoch in range(args.epochs+1):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(G, features)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float())
    acc_train = accuracy(output[idx_sens_train], sens[idx_sens_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(G, features)
    if epoch%10==0:
        roc_test = f1_score(sens[idx_test].cpu().numpy(),output[idx_test].detach().cpu().numpy()>0)
        loss_test = F.binary_cross_entropy_with_logits(output[idx_test], sens[idx_test].unsqueeze(1).float())
        acc_test = accuracy(output[idx_test], sens[idx_test])
        print("Epoch [{}] Test set results:".format(epoch),
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
            "ROC: {:.4f}".format(roc_test))
        if acc_test > best_roc:
            best_roc = acc_test
            torch.save(model.state_dict(),"./checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number))
        print(best_roc)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# %%
