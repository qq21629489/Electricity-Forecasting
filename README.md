# DSAI-HW-2021
* 製造所 P96104112 蘇冠瑜
* 製造所 P96101148 巫清賢

## 第一次實驗
* 模型：Singal/Multi-step LSTM model ![image](https://github.com/qq21629489/Electricity-Forecasting/blob/main/picture/LSTM%20model%20arch.png)
* 特徵：原始資料（based data） + dayofweek（星期幾，用0～6表示）
* 結果：這次實驗了單層、雙層LSTM，並且多加入星期的特徵，但兩種結果上MAE依然來到了600以上，效果不是很好。另外在搭建上也不是非常方便，因此嘗試第二次實驗。![image](https://github.com/qq21629489/Electricity-Forecasting/blob/main/picture/LSTM%20mae.png)
* 其他：
    * nn size：100
    * batch size：128
    * epoch：100
    * loss：![image](https://github.com/qq21629489/Electricity-Forecasting/blob/main/picture/LSTM%20loss.png)

## 第二次實驗
* 模型：FbProphet ![image](https://github.com/qq21629489/Electricity-Forecasting/blob/main/picture/fpprohent%20model%20arch.png)
* 特徵：原始資料（based data）中的日期、備轉容量（MW）
* 結果：prophet為單特徵訓練模型，預測的結果比起第一次實驗使用LSTM還要穩定的，並且，MAE比起第一次實驗降低許多。![image](https://github.com/qq21629489/Electricity-Forecasting/blob/main/picture/fpprohent%20predict.png)

## 備註
1. 若使用requirement.txt後，依然有版本問題或其他bug，可以嘗試使用`backup/requirement.txt`，這個檔案是使用conda自動生成的，各套件版本理論上是沒有問題。