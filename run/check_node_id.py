# 读取文件
with open('/root/autodl-tmp/demo/roland_public_data/multi_social_networks/MLKing2013/new_id.txt', 'r') as f:
    lines = f.readlines()

# 提取节点ID
nodes = set()
for line in lines:
    parts = line.split()
    nodes.add(int(parts[0]))  # 获取节点ID
    nodes.add(int(parts[1]))  # 获取节点ID


# 计算节点个数和节点ID范围
node_list = sorted(nodes)
node_count = len(node_list)
min_node = node_list[0]
max_node = node_list[-1]

# 判断节点ID是否连续
is_continuous = (max_node - min_node + 1 == node_count)

# 输出结果
print(f"节点个数: {node_count}")
print(f"节点ID范围: {min_node} 到 {max_node}")
print(f"节点ID是否连续: {'是' if is_continuous else '否'}")
