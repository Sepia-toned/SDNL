import networkx as nx
import numpy as np

def centrality_fun(graph, **kwargs):
    centrality = nx.betweenness_centrality(graph)
    return [centrality[x] for x in graph.nodes()]

def pagerank_fun(graph, **kwargs):
    # TODO multiple feature dimensions
    pagerank = nx.pagerank(graph)
    return [pagerank[x] for x in graph.nodes]

def clustering_coefficient_fun(graph, **kwargs):
    return list(nx.clustering(graph).values())

data_path = '/root/autodl-tmp/demo/RT_friend/sub_friend.txt'
graph = nx.read_edgelist(data_path, create_using=nx.DiGraph())

nodes_list = graph.nodes
in_degree = [item[1] for item in graph.in_degree()]
out_degree = [item[1] for item in graph.out_degree()]
pagerank = pagerank_fun(graph)
cluster = clustering_coefficient_fun(graph)
# centrality = centrality_fun(graph)

save_path = '/root/autodl-tmp/demo/RT_friend/static_feature.txt'

# 将列表转换为NumPy数组
node_list_arr = np.array(nodes_list)
in_degree_arr = np.array(in_degree)
out_degree_arr = np.array(out_degree)
pagerank_arr = np.array(pagerank)
cluster_arr = np.array(cluster)
# centrality_arr = np.array(centrality)

# 将数组合并为一个二维数组
data_matrix = np.column_stack((node_list_arr, in_degree_arr, out_degree_arr, pagerank_arr, cluster_arr))

# 保存二维数组到文件
np.savetxt(save_path, data_matrix, fmt='%s', delimiter='\t')
