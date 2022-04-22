# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:29:37 2022

@author: serca
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import json
from Strategy_3 import trade
import warnings

warnings.filterwarnings("ignore")
f_1h = open('Binance_1h_BTCUSDT_Candles.json',)
data_bin_1h = json.load(f_1h)
df_1h = pd.DataFrame(data_bin_1h,columns=["Open time","Open","High","Low", "Close","Volume", 
                                                  "Close time","Quote asset volume","Number of trades", 
                                                  "Taker buy base asset volume", "Taker buy quote asset volume","Ignore" ]).astype(float)
df_1h = df_1h.set_index("Close time")
df_1h = df_1h.drop(columns = ["Ignore","Taker buy base asset volume", "Taker buy quote asset volume" ])
start_time_1h = df_1h.index[0]
end_time_1h = df_1h.index[-1]
day = 86400000


def Trade_Analysis():
    pd.set_option('display.max_columns', None)
    #print((trade(df_1h, end_time_1h-200*day, end_time_1h - 100*day, 280, True, True, True, 0.99925)[0]))
    trades_df = (trade(df_1h, end_time_1h - 100*day, end_time_1h - 1*day ,270, True, True, False, 0.99925)[1])
    trades_df = trades_df[trades_df["Win Situation"].notna()]
    trades_df['L.T. Time in Trade'] = trades_df['Time in Trade'].shift(+1)
    trades_df['L.T. Win Situation'] = trades_df['Win Situation'].shift(+1)
    trades_df['L.T. Return on Trade'] = trades_df['Return on Trade'].shift(+1)
    trades_df['L.T. Max Drawdown'] = trades_df['Max Drawdown'].shift(+1)
    trades_df['L.T. Max Drawup'] = trades_df['Max Drawup'].shift(+1)
   
    
    # Adding win sequences
    losing_indexes = (trades_df[trades_df["Win Situation"] == -1].index.values.tolist())
    win_sequences = []
    if trades_df.index.values.tolist()[0] not in losing_indexes:
        item = ((trades_df.loc[trades_df.index.values.tolist()[0]:losing_indexes[0]]["Win Situation"].cumsum()).values.tolist())
        win_sequences.append(item[ : -1])
    for i in range(0,len(losing_indexes)-1): 
        
        item = ((trades_df.loc[losing_indexes[i]:losing_indexes[i+1]]["Win Situation"].cumsum()+1).values.tolist())
        #item[0] += 1
        win_sequences.append(item[ : -1])
    
    if len(win_sequences)+1 == len(trades_df):
        win_sequences.append(0)
    elif len(win_sequences) < len(trades_df):
        item = ((trades_df.loc[losing_indexes[-1]:trades_df.index.values.tolist()[-1]]["Win Situation"].cumsum()+1).values.tolist())
        #item[0] += 1
        win_sequences.append(item)
    win_sequences = [item for sublist in win_sequences for item in sublist]
    trades_df["Win Sequence"] = win_sequences
    trades_df['Win Sequence'] = trades_df['Win Sequence'].shift(+1)
    ############

    
    short_trades = trades_df.copy(deep = True)
    short_trades= short_trades[short_trades["Sell Points"].notna()]
    #print("Sell Trades df:   \n",sell_trades)
    winning_short_trades = short_trades[short_trades["Win Situation"] == 1]
    losing_short_trades = short_trades[short_trades["Win Situation"] == -1]
    # print(winning_short_trades[["Sell Volumes", "Sell Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup"]])
    # print(losing_short_trades[["Sell Volumes", "Sell Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup"]])
    
    long_trades = trades_df.copy(deep = True)
    long_trades= long_trades[long_trades["Buy Points"].notna()]
    #print("Buy Trades df:   \n",buy_trades)
    winning_long_trades = long_trades[long_trades["Win Situation"] == 1]
    losing_long_trades = long_trades[long_trades["Win Situation"] == -1]
    #print(winning_long_trades[["Buy Volumes", "Buy Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup", "Volumes in Trade"]])
    #print(losing_long_trades[["Buy Volumes", "Buy Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup", "Volumes in Trade"]])
    #trades_df["Negative Time in Trade"] = np.nan
    #trades_df["Negative Time in Trade"] = trades_df["Time in Trade"] * trades_df["Win Situation"]

# =============================================================================
#     plt.style.use('seaborn-bright')
#     plt.figure(figsize= (120,15))
#     plt.hist(trades_df["Negative Time in Trade"], bins=np.arange(min(trades_df["Negative Time in Trade"]), max(trades_df["Negative Time in Trade"]) + 1, 1))
#     plt.show()
# =============================================================================
    trades_df = trades_df.fillna(0)

    trades_df.to_csv("Test_Trades_DF.csv")
    short_trades.to_csv("Test_Short_Trades_DF.csv")
    long_trades.to_csv("Test_Long_Trades_DF.csv")

    return trades_df, long_trades, short_trades

Trade_Analysis()

#k.loc[losing_indexes[i]:losing_indexes[i+1]]["Wi]n Sequence"] = 
