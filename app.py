# system
import os

# basic
import numpy as np
import pandas as pd

# visual
import matplotlib.pyplot as plt

# Prophet
from fbprophet import Prophet

# sklearn evaluate
from sklearn import metrics

def createData(file_name='training_data.csv'):
    print('creating data...')
    
    df1 = pd.read_csv('台灣電力公司_過去電力供需資訊.csv')
    df1['日期'] = pd.to_datetime(df1['日期'], format='%Y%m%d')
    df1 = df1.loc[:, ['日期', '備轉容量(MW)']]

    df2 = pd.read_csv('本年度每日尖峰備轉容量率.csv')
    df2['日期'] = pd.to_datetime(df2['日期'], format='%Y/%m/%d')
    df2 = df2.loc[:, ['日期', '備轉容量(萬瓩)']]
    df2['備轉容量(萬瓩)'] = df2['備轉容量(萬瓩)'].map(lambda x: int(x*10))
    df2 = df2.rename(columns={'備轉容量(萬瓩)': '備轉容量(MW)'})
    df2 = df2[df2['日期'] >= '2022-03-01']

    df3 = pd.concat([df1, df2], ignore_index=True)
    df3.to_csv(file_name, index=False)
    
    print('{} created.'.format(file_name))

def loadData(file_name='training_data.csv'):
    if os.path.isfile(file_name) == False:
        createData(file_name)
        
    df = pd.read_csv(file_name)
    return df

def predict(df):
    print('start predict...')
    
    # create model of prophet
    model = Prophet()

    # reset columns to ds(date), y(data)
    new_df = df.rename(columns={'日期':'ds', '備轉容量(MW)':'y'})

    # training model
    model.fit(new_df)

    # create predict time series
    future = model.make_future_dataframe(periods=30)

    # start predict
    forecast = model.predict(future)
    
    # display
    # figure = model.plot(forecast)
    # plt.show()
    
    print('predict done.')
    
    # modify date range and rename columns
    result_df = pd.DataFrame(forecast)
    
    result_df = result_df.loc[:, ['ds', 'yhat']]
    
    result_df = result_df[result_df['ds'] >= '2022-03-30']
    result_df = result_df[result_df['ds'] <= '2022-04-13']
    
    result_df['ds'] = result_df['ds'].dt.strftime('%Y%m%d')
    result_df['yhat'] = result_df['yhat'].apply(lambda x: int(x))
    
    result_df = result_df.rename(columns={'ds':'date', 'yhat':'operating_reserve(MW)'})
    
    return result_df

def createSubmisson(df : pd.DataFrame, file_name='submission.csv'):
    print('creating submission...')
    
    df.to_csv(file_name, index=False)
    
    print('{} created.'.format(file_name))

# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()


    # The following part is an example.
    # You can modify it at will.

    df = loadData(args.training)
    
    result_df = predict(df)
    
    createSubmisson(result_df, args.output)