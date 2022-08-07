"""
We implement the strategy employed by "Factorized Graph Representations forSemi-Supervised Learning from Sparse Data" (Kumar et al.)
in order to obtain the compatibility matrix for belief propagation algorithms

We will only use features from nodes marked as "train". 
If a node is not marked as train, we will set its label value to 0.
"""

import dgl
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

if __name__ == "__main__":
	LEARNING_RATE 	= 1e-6
	EARLY_TERMINATE = 3

	dataset 		= Planetoid(root='/tmp/pubmed', name='Pubmed')[0]
	num_classes 	= dataset.y.max() +1
	feats 			= dataset.y * (dataset.val_mask == False)
	feats_1_hot 	= nn.functional.one_hot(feats, num_classes = num_classes).type(torch.FloatTensor)
	
	adj_mat 		= to_scipy_sparse_matrix(dataset.edge_index)
	adj_mat 		= dgl.from_scipy(adj_mat).adjacency_matrix()

	adj_mat_feats 	= torch.matmul(adj_mat, feats_1_hot)
	H_trainable 	= torch.rand((num_classes -1, num_classes -1), requires_grad = True)

	loss_fnct 		= nn.MSELoss(reduction = "mean")
	min_loss 		= sys.maxsize
	loss_increasing_count = 0

	for _ in range(1000):
		H_full 				= F.pad(input = H_trainable, pad=(0, 1, 0, 1), mode = 'constant', value = 0)
		H_full[:-1, -1] 	= 1 - torch.sum(H_trainable, dim = 1)
		H_full[-1, :-1] 	= 1 - torch.sum(H_trainable, dim = 0)
		H_full[-1, -1] 		= 1 - torch.sum(H_full[:-1, -1], dim = 0)

		loss = loss_fnct(feats_1_hot, torch.matmul(adj_mat_feats, H_full))
		loss.backward()
		print(loss)

		if loss < min_loss:
			min_loss 	= loss 
			min_H 		= H_full
			loss_increasing_count = 0

		else:
			loss_increasing_count += 1

		with torch.no_grad():
			H_trainable -= LEARNING_RATE * H_trainable.grad

		if loss_increasing_count >= EARLY_TERMINATE:
			break
	
	print(min_loss, min_H)
	torch.save(min_H, os.path.join(os.getcwd(), "matrices", "pubmed", "H.pt"))