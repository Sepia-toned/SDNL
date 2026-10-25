import os
import pandas as pd

# 设置文件夹路径
folder_path = '/root/autodl-tmp/demo/run/otc_gnn_layer_ex_5/'

# 创建三个 10x10 的空 DataFrame，分别用于保存每个时间步的 ACC 数据
acc_table_t1 = pd.DataFrame(index=range(1, 11), columns=range(1, 11))  # Time step 1
acc_table_t2 = pd.DataFrame(index=range(1, 11), columns=range(1, 11))  # Time step 2
acc_table_t3 = pd.DataFrame(index=range(1, 11), columns=range(1, 11))  # Time step 3

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理以 .xlsx 结尾的 Excel 文件
    if filename.endswith('AUC.xlsx'):
        # 提取文件名中的 num1 和 num2
        parts = filename.split('_')
        num1 = int(parts[0][2:])  # dy{num1}
        num2 = int(parts[1][2:])  # st{num2}
        
        # 读取 Excel 文件中的数据
        file_path = os.path.join(folder_path, filename)
        data = pd.read_excel(file_path, header=None)
        
        # 提取 ACC 数据，假设数据位于第二到第四行
        acc_data = data.iloc[1:4, 0].values  # 取第 2 到 4 行的数据

        # 将 ACC 数据分别填入对应时间步的表格中
        acc_table_t1.at[num1, num2] = acc_data[0]  # 时间步1
        acc_table_t2.at[num1, num2] = acc_data[1]  # 时间步2
        acc_table_t3.at[num1, num2] = acc_data[2]  # 时间步3

# 如果需要保存为新的 Excel 文件
output_file = os.path.join(folder_path, 'auc_tables.xlsx')
with pd.ExcelWriter(output_file) as writer:
    acc_table_t1.to_excel(writer, sheet_name='Time Step 1')
    acc_table_t2.to_excel(writer, sheet_name='Time Step 2')
    acc_table_t3.to_excel(writer, sheet_name='Time Step 3')

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理以 .xlsx 结尾的 Excel 文件
    if filename.endswith('ACC.xlsx'):
        # 提取文件名中的 num1 和 num2
        parts = filename.split('_')
        num1 = int(parts[0][2:])  # dy{num1}
        num2 = int(parts[1][2:])  # st{num2}
        
        # 读取 Excel 文件中的数据
        file_path = os.path.join(folder_path, filename)
        data = pd.read_excel(file_path, header=None)
        
        # 提取 ACC 数据，假设数据位于第二到第四行
        acc_data = data.iloc[1:4, 0].values  # 取第 2 到 4 行的数据

        # 将 ACC 数据分别填入对应时间步的表格中
        acc_table_t1.at[num1, num2] = acc_data[0]  # 时间步1
        acc_table_t2.at[num1, num2] = acc_data[1]  # 时间步2
        acc_table_t3.at[num1, num2] = acc_data[2]  # 时间步3

# 如果需要保存为新的 Excel 文件
output_file = os.path.join(folder_path, 'acc_tables.xlsx')
with pd.ExcelWriter(output_file) as writer:
    acc_table_t1.to_excel(writer, sheet_name='Time Step 1')
    acc_table_t2.to_excel(writer, sheet_name='Time Step 2')
    acc_table_t3.to_excel(writer, sheet_name='Time Step 3')

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理以 .xlsx 结尾的 Excel 文件
    if filename.endswith('AP.xlsx'):
        # 提取文件名中的 num1 和 num2
        parts = filename.split('_')
        num1 = int(parts[0][2:])  # dy{num1}
        num2 = int(parts[1][2:])  # st{num2}
        
        # 读取 Excel 文件中的数据
        file_path = os.path.join(folder_path, filename)
        data = pd.read_excel(file_path, header=None)
        
        # 提取 ACC 数据，假设数据位于第二到第四行
        acc_data = data.iloc[1:4, 0].values  # 取第 2 到 4 行的数据

        # 将 ACC 数据分别填入对应时间步的表格中
        acc_table_t1.at[num1, num2] = acc_data[0]  # 时间步1
        acc_table_t2.at[num1, num2] = acc_data[1]  # 时间步2
        acc_table_t3.at[num1, num2] = acc_data[2]  # 时间步3

# 如果需要保存为新的 Excel 文件
output_file = os.path.join(folder_path, 'ap_tables.xlsx')
with pd.ExcelWriter(output_file) as writer:
    acc_table_t1.to_excel(writer, sheet_name='Time Step 1')
    acc_table_t2.to_excel(writer, sheet_name='Time Step 2')
    acc_table_t3.to_excel(writer, sheet_name='Time Step 3')