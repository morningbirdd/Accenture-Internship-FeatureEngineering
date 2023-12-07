import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
df = pd.read_csv(r'D:\Desktop\个人\实习\埃森哲\定价分析\定价分析数据预处理.csv')
#定义无用需删除的特征
drop_feature=['compressionratio','peakrpm','stroke','carheight','carbody', 
              'enginelocation','fuelsystem','enginetype','drivewheels',
              'fueltypes','name','aspiration']
#删除特征
df=df.drop(drop_feature,axis=1)
#查看数据信息
print(df.info())
#将删除特征后的数据保存在同一目录下并命名为“删除特征后的数据.csv”
df.to_csv(r'D:\Desktop\个人\实习\埃森哲\定价分析\删除特征后的数据.csv',index=False,sep=',')




