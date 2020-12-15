import tushare as ts
import matplotlib.pyplot as plt


df=ts.get_k_data('600519',ktype='D',start='2010-04-26',end='2020-04-26')
data_path="./SH600519.csv"
df.to_csv(data_path)