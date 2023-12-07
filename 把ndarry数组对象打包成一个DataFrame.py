import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('D:/Desktop/个人/实习/埃森哲/定价分析/定价分析数据预处理.csv')
# 定义数值变量和类别变量
numeric_vars = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'curbweight',
                'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg']
categorical_vars = ['fueltypes', 'aspiration', 'carbody', 'drivewheels', 'enginelocation', 'enginetype', 'fuelsystem']
#对数值型数据进行标准化
#初始标准化器
scaler = StandardScaler()
#对数值型数据进行标准化
df[numeric_vars] = scaler.fit_transform(df[numeric_vars])
## 将ndarray数组对象包装成DataFrame对象
df=pd.DataFrame(df,columns=numeric_vars+categorical_vars )
#打印出新的数据集的信息
print(df.info())
