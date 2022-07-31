import numpy as np
import os
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from utils.GraphConverter import GraphConverter

if __name__ == "__main__":
	plaintoid_g = Planetoid(root='/tmp/pubmed', name='Pubmed')[0]
	edge_H 		= torch.load(os.path.join(os.getcwd(), "matrices", "pubmed", "H.pt"))
	edge_H 		= edge_H.detach().numpy()

	observed_nodes 		= torch.where(plaintoid_g.val_mask == False)[0].tolist()
	observed_node_class = plaintoid_g.y[plaintoid_g.val_mask == False].tolist()

	factor_g 	= GraphConverter.planetoid_to_factor_graph(	plaintoid_g 		= plaintoid_g, 
															num_classes 		= plaintoid_g.y.max() +1,
															edge_H 				= edge_H,
															observed_nodes 		= observed_nodes,
															observed_node_class = observed_node_class
														)

	iters, converged = factor_g.lbp(normalize = True, max_iters = 100)
	print(f"Iters: {iters}, Converged {converged}")

	marginals 				= factor_g.rv_marginals()
	unobserved_nodes 		= torch.where(plaintoid_g.val_mask == True)[0].tolist()
	
	unobserved_node_class 	= plaintoid_g.y[plaintoid_g.val_mask == True]
	predicted_class 		= filter(lambda x: int(x[0].name) in unobserved_nodes, marginals)
	predicted_class 		= list(map(lambda x: x[-1], predicted_class))
	predicted_class 		= torch.tensor(predicted_class)
	loss_fnct 				= nn.CrossEntropyLoss(reduction = "mean")
	loss = loss_fnct(predicted_class, unobserved_node_class)
	print(loss)