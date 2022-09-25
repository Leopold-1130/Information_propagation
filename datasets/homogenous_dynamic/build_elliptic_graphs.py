import networkx as nx
import pandas as pd

if __name__ == "__main__":
	node_class_df 	= pd.read_csv("/Users/johan.kok/Downloads/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
	edges_df 		= pd.read_csv("/Users/johan.kok/Downloads/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
	features_df 	= pd.read_csv("/Users/johan.kok/Downloads/elliptic_bitcoin_dataset/elliptic_txs_features.csv", header = None, names = ["node_id", "timestamp"] + [f"feat_{i}" for i in range(165)])

	global_G	= nx.Graph()

	for _, row in node_class_df.iterrows():
		tx_cls 	= row["class"]
		tx_cls 	= int(tx_cls) if tx_cls != "unknown" else 0
		global_G.add_node(row["txId"], cls = tx_cls)

	for _, row in edges_df.iterrows():
		global_G.add_edge(row["txId1"], row["txId2"])

	for ts in range(45, 50):
		nodes_at_ts = features_df[features_df.timestamp == ts].node_id.tolist()
		sub_G 		= global_G.subgraph(nodes_at_ts)
		nx.write_gpickle(sub_G, f"elliptic_{ts}.gpickle")