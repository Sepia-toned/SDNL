import os
import pandas as pd
import numpy as np

# 输入文件夹路径（包含gnn_layer_ex_1到gnn_layer_ex_5）
input_base_dir = '/root/autodl-tmp/demo/run'
output_dir = '/root/autodl-tmp/demo/run/otc_gnn_layer_ex_avg'

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义要处理的文件名
files = ['acc_tables.xlsx', 'ap_tables.xlsx', 'auc_tables.xlsx']

# 用于存储平均值的字典
averages = {file: [] for file in files}

# 遍历文件夹gnn_layer_ex_1到gnn_layer_ex_5
for i in range(1, 6):
    folder = os.path.join(input_base_dir, f'otc_gnn_layer_ex_{i}')
    
    # 读取每个表格并计算平均
    for file in files:
        file_path = os.path.join(folder, file)
        
        # 如果文件存在
        if os.path.exists(file_path):
            print(f"正在处理文件: {file_path}")
            
            # 获取Excel文件中的所有工作表名称
            sheet_names = pd.ExcelFile(file_path).sheet_names
            print(f"可用的工作表: {sheet_names}")
            
            # 初始化一个列表存储该文件中所有sheet的值
            sheet_values = []
            
            # 遍历所有sheet
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    # 将每个sheet的数据转化为numpy数组
                    sheet_values.append(df.values)
                except Exception as e:
                    print(f"无法处理工作表 {sheet}：{e}")
            
            # 将该文件在所有文件夹中的sheet值存入字典
            if sheet_values:
                averages[file].append(sheet_values)

# 对每个表格在5个文件夹中的结果进行逐位置平均
final_results = {}
for file, all_sheets in averages.items():
    avg_sheets = []
    
    # 对每个sheet进行逐位置平均
    for sheet_idx in range(len(all_sheets[0])):  # 假设每个文件夹中的sheet数量相同
        # 获取所有文件夹该sheet的所有数据
        sheet_values = np.array([sheet[sheet_idx] for sheet in all_sheets])
        
        # 对应位置的值进行平均
        avg_sheet = sheet_values.mean(axis=0)
        avg_sheets.append(avg_sheet)
    
    # 保存每个表格的结果
    final_results[file] = avg_sheets

# 保存结果
for file, result in final_results.items():
    with pd.ExcelWriter(os.path.join(output_dir, file)) as writer:
        for idx, sheet in enumerate(result):
            sheet_df = pd.DataFrame(sheet)
            sheet_df.to_excel(writer, sheet_name=f'Time Step {idx+1}', index=False)

print("处理完成！")
