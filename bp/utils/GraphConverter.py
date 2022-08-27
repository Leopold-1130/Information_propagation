import factorgraph as fg
import numpy as np

class GraphConverter(object):
	@classmethod
	def load_graph(cls, json_name):
	    """

	    Args:
	        json_name: json file name

	    Returns:
	        networkx graph
	    """
	    from networkx.readwrite import json_graph
	    import json
	    with open(json_name, 'r') as f:
	        data = json.load(f)
	    f.close()
	    graph = json_graph.node_link_graph(data)
	    print('Loaded {} into the nextorkx graph'.format(json_name))
	    return graph

	@classmethod
	def planetoid_to_factor_graph(self, plaintoid_g, 
										num_classes: int, 
										edge_H: np.array,
										observed_nodes: [int],
										observed_node_class: [int],
								):
		edges 		= plaintoid_g.edge_index
		num_nodes 	= plaintoid_g.size()[0]
		factor_g 	= fg.Graph()

		for i in range(num_nodes):
			factor_g.rv(str(i), num_classes)

		for col in range(plaintoid_g.edge_index.shape[-1]):
			[node_A, node_B] = plaintoid_g.edge_index[:, col]
			node_A 	= node_A.item()
			node_B 	= node_B.item()
			factor_g.factor([str(node_A), str(node_B)], potential = edge_H)

		for each_observed_node, each_observed_node_class in zip(observed_nodes, observed_node_class):
			factor_g.factor([str(each_observed_node)], potential = np.array([1.0 if i == each_observed_node_class else 0.0 for i in range(num_classes)]))

		return factor_g

	@classmethod
	def yelp_to_factor_graph(cls, 	yelp_g, 
									user_review_H, 
									review_user_H,
									product_review_H, 
									review_product_H, 
								):

		factor_g 	= fg.Graph()

		for each_node in yelp_g.nodes():
			node_prior = yelp_g.nodes()[each_node]["prior"]
			factor_g.rv(each_node, 2)
			factor_g.factor([each_node], potential = np.array([1 - node_prior, node_prior]))

		for (from_vertice, to_vertice) in yelp_g.edges():
			edge_name 			= f"{from_vertice}_{to_vertice}"
			from_vertice_type 	= yelp_g.nodes()[from_vertice]["types"]
			to_vertice_type 	= yelp_g.nodes()[to_vertice]["types"]
			review_prior 		= yelp_g.edges()[(from_vertice, to_vertice)]["prior"]

			factor_g.rv(edge_name, 2)
			factor_g.factor([edge_name], potential = np.array([1 - review_prior, review_prior]))
			
			if from_vertice_type == "user" and to_vertice_type == "prod":
				factor_g.factor([from_vertice, edge_name], potential = user_review_H)
				factor_g.factor([edge_name, to_vertice], potential = review_product_H)

			else:
				factor_g.factor([from_vertice, edge_name], potential = product_review_H)
				factor_g.factor([edge_name, to_vertice], potential = review_user_H)

		return factor_g