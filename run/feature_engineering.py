import numpy as np
import torch

def fun():
    feature_path = '/root/autodl-tmp/demo/RT_friend/static_feature.txt'
    num_nodes = 256491
    feature_data = []
    with open(feature_path, "r") as feature_file:
        for line in feature_file:
            items = line.strip().split()
            items = [float(item) for item in items]
            feature_data.append(items)

    # 将列表转换为二维tensor
    feature_tensor = torch.tensor(feature_data)
    complete_column = np.arange(num_nodes)

    # 找出缺失的元素
    missing_elements = np.setdiff1d(complete_column, feature_tensor[:, 0])

    # 对于缺失的元素,将其插入到feature_tensor的第一列,并将后面的元素设置为0
    for element in missing_elements:
        new_row = np.zeros(feature_tensor.shape[1])
        new_row[0] = element
        feature_tensor = np.vstack([feature_tensor, new_row])

    # 按照第一列的值进行排序
        feature_tensor = feature_tensor[feature_tensor[:, 0].argsort()]
    
    feature_tensor = feature_tensor[:, 1:]
    save_path = '/root/autodl-tmp/demo/RT_friend/A_feature.txt'
    with open(save_path, 'w') as file:
        for row in feature_tensor:
        # 将每行数据转换为字符串并以空格分隔
            row_str = ' '.join([str(elem.item()) for elem in row])
            file.write(row_str + '\n')

fun()