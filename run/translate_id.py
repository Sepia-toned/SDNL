# 读取原始文件
with open('/root/autodl-tmp/demo/roland_public_data/multi_social_networks/HiggsTwitter/activity.txt', 'r') as f:
    lines = f.readlines()

# 提取所有唯一的节点ID
nodes = set()
for line in lines:
    parts = line.split()
    nodes.add(int(parts[0]))  # 获取源节点ID
    nodes.add(int(parts[1]))  # 获取目标节点ID

# 获取节点的数量
node_count = len(nodes)
print(f"节点个数: {node_count}")

# 创建映射：原节点ID -> 新节点ID（从0到node_count-1）
sorted_nodes = sorted(nodes)  # 排序所有节点
node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}  # 映射原ID到新的连续ID

# 将映射后的ID保存到新文件
with open('/root/autodl-tmp/demo/roland_public_data/multi_social_networks/HiggsTwitter/new_id.txt', 'w') as f:
    for line in lines:
        parts = line.split()
        # 使用映射更新源节点和目标节点ID
        source_node = node_mapping[int(parts[0])]
        target_node = node_mapping[int(parts[1])]
        # 写入新内容：更新后的源节点、目标节点、时间戳和网络层
        f.write(f"{source_node} {target_node} {parts[2]} {parts[3]}\n")

print("节点ID已更新并保存至 'new_id.txt' 文件")
