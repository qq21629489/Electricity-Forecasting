import pandas as pd

# 載入訓練資料
df1 = pd.read_csv('台灣電力公司_過去電力供需資訊.csv')
df1['日期'] = pd.to_datetime(df1['日期'], format='%Y%m%d')

#  ,'備轉容量率(%)','工業用電(百萬度)','民生用電(百萬度)','民生用電(百萬度)','核一#1(萬瓩)','核一#2(萬瓩)','核二#1(萬瓩)','核二#2(萬瓩)','核三#1','核三#2','林口#1','林口#2','林口#3','台中#1','台中#2','台中#3','台中#4','台中#5','台中#6','台中#7','台中#8','台中#9','台中#10'
df1 = df1.loc[:, ['日期', '備轉容量(MW)']]
# 把日期設定成y軸
# d1 = df1.set_index('日期')
print(df1)
print('='*20)


df2 = pd.read_csv('本年度每日尖峰備轉容量率.csv')
df2['日期'] = pd.to_datetime(df2['日期'], format='%Y/%m/%d')
df2 = df2.loc[:, ['日期', '備轉容量(萬瓩)']]
df2['備轉容量(萬瓩)'] = df2['備轉容量(萬瓩)'].map(lambda x: int(x*10))
df2 = df2.rename(columns={'備轉容量(萬瓩)': '備轉容量(MW)'})
df2 = df2[df2['日期'] >= '2022-03-01']
# d2 = df2.set_index('日期')
print(df2)
print('='*20)

df3 = pd.concat([df1, df2], ignore_index=True)
print(df3)
df3.to_csv('123.csv')