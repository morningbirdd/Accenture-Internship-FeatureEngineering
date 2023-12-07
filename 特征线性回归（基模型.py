import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 使用黑体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  
# 读取数据
df = pd.read_csv('D:/Desktop/个人/实习/埃森哲/定价分析/定价分析数据预处理.csv')

# 定义数值变量和类别变量
numeric_vars = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'curbweight',
                'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg']
categorical_vars = ['fueltypes', 'aspiration', 'carbody', 'drivewheels', 'enginelocation', 'enginetype', 'fuelsystem']
#####################
# 建立基模型
X = df[['enginesize']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('线性模型（基模型）的均方误差：', mean_squared_error(y_test, y_pred))

# 使用所有数值变量进行拟合
X = df[numeric_vars]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('使用所有数值变量的线性模型的均方误差：', mean_squared_error(y_test, y_pred))
# 权重可视化
plt.figure(figsize=(10, 5))
sns.barplot(x=model.coef_, y=numeric_vars)
plt.xlabel('权重')
plt.ylabel('特征')
plt.title('特征权重&线性回归')
plt.show()
