import numpy as np
import os
import torch
import torch.nn as nn
from utils.GraphConverter import GraphConverter
from utils.helper import *
from random import sample

if __name__ == "__main__":
	file_name 	= '../datasets/heterogeneous_static/Yelp_graph_data.json'
	yelp_g 		= GraphConverter.load_graph(json_name = file_name)
	user_ground_truth = node_attr_filter(yelp_g, 'types', 'user', 'label')
	
	user_review_potential 		= np.array([[0.2, 0.8], [0.8, 0.2]])
	review_product_potential 	= np.array([[0.1, 0.9], [0.9, 0.1]])

	yelp_fg 	= GraphConverter.yelp_to_factor_graph(	yelp_g = yelp_g, 
														user_review_H = user_review_potential, 
														review_user_H = user_review_potential,
														product_review_H = review_product_potential, 
														review_product_H = review_product_potential
													)


	iters, converged = yelp_fg.lbp(normalize = True, max_iters = 15)
	print(f"Iters: {iters}, Converged {converged}")

	marginals 				= yelp_fg.rv_marginals()
	marginals_dict 			= { str(node) : pred for (node, pred) in marginals }

	cls_1_users 			= list(filter(lambda x: x[1] == 1, user_ground_truth.items()))
	cls_0_users 			= list(filter(lambda x: x[1] == 0, user_ground_truth.items()))

	test_1_nodes 			= list(map(lambda x: x[0], sample(cls_1_users, int(0.05 * len(user_ground_truth)))))
	test_0_nodes 			= list(map(lambda x: x[0], sample(cls_0_users, int(0.05 * len(user_ground_truth)))))
	test_nodes 				= test_1_nodes + test_0_nodes

	pred_dist 	= []
	truth_cls 	= []
	
	for each_test_node in test_nodes:
		pred_dist.append(marginals_dict[each_test_node])
		truth_cls.append(yelp_g.nodes()[each_test_node]["label"])

	truth_tensor 			= torch.tensor(truth_cls)
	_, best_predicted_class = torch.max(torch.tensor(pred_dist), dim = 1)
	accuracy 				= torch.sum(best_predicted_class == truth_tensor) / len(truth_tensor)
	print(f"Accuracy {accuracy}")