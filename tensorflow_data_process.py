import os
import pandas as pd
import tensorflow as tf

os.makedirs(os.path.join('..', 'data/tensorflow'), exist_ok=True)
data_file = os.path.join('..', 'data/tensorflow', 'house_tiny.csv')

# 写数据
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行数据
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取数据
data = pd.read_csv(data_file)
print(data)

# 处理缺失值
# iloc 位置索引
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)

# fillna 处理缺失值
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

# 独热编码
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)


x, y = tf.constant(inputs.values), tf.constant(outputs.values)
print(x, y)
