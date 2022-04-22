# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 03:15:58 2022

@author: serca
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import json

#double_ma_trade(df, x, y, start_time, end_time, mov_avg_type, long, short, show_on_graph, fee, ema_fisher_period): 

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
def fisher_transform(df, period, show_on_graph):
    fisher_liste = []
    fisher_liste_ma = []
        
    for i in range(0, len(df)):
        if i< period-1:
            fisher_liste.append(np.nan)
            fisher_liste_ma.append(np.nan)
        else:
            data_list = df.iloc[i-period+1 : i+1, :  ]["Close"]
            minus1= min(data_list)
            max1 = max(data_list)
            value = df.iloc[i , :  ]["Close"]
            a = 1 - 2*(max1-value)/(max1-minus1)
            r = 2*(value-minus1)/(max1-minus1) -1
            if r == 1:
                r = 0.999
            elif r == -1:
                r = -0.999
            fish = (np.log((1+r)/(1-r)))*0.5
            fisher_liste.append(fish) 

        
    
    #fisher_liste = pd.Series(fisher_liste)
    df["Fisher"] = fisher_liste
    df["Fisher MA"] = df["Fisher"].rolling(window = 7).mean() 
    
    
    if show_on_graph:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(df["Fisher"], linewidth = 3, alpha = 0.8)
        plt.plot(df["Fisher MA"], linewidth = 3, alpha = 0.8)
        plt.grid()
    return df
def ema(values, period):
    return values.ewm(span=period).mean()

def trade(orig_df, start_time, end_time, ema_period, long, short, show_on_graph, fee,  time_in_trade_limit, leverage, sequence_limit):
                                                                                     #ema p. # ma. #fisher ema per.  

    df = orig_df.copy(deep = True)
    index1 = "EMA" + str(ema_period)
    df[index1] = ema(df["Close"], ema_period)
    

    cross_df = pd.DataFrame()
    cross_df["Close"] = df.loc[start_time : end_time, "Close"]
    #cross_df["Fisher"] = df.loc[start_time : end_time, "Fisher"]
    #cross_df["RSI"] = rsi_
    cross_df[index1] = df.loc[start_time : end_time, index1]
    cross_df["Volume"] = df.loc[start_time : end_time, "Volume"]
    
    wallet = 10000
    buy_points = []
    sell_points = []

    cond = 0
    first_changer = 0
    time_in_trade = 1

    reverse_strategy = 0
    num_of_trades = 0
    num_of_wins = 0
    length_of_win_sequence = 0
    
    
    for i in range(0,len(cross_df)):
        
        
        sequence_alarm = length_of_win_sequence>= sequence_limit
#double_ma_trade(df, x, y, start_time, end_time, mov_avg_type, long, short, show_on_graph, fee, ema_fisher_period): 
        crossover = ((cross_df.iloc[i-1,]["Close"]< cross_df.iloc[i-1,][index1]) and  (cross_df.iloc[i,]["Close"]> cross_df.iloc[i,][index1])) or ((cross_df.iloc[i-1,]["Close"]> cross_df.iloc[i-1,][index1]) and  (cross_df.iloc[i,]["Close"]< cross_df.iloc[i,][index1]))
        if reverse_strategy == 0 and crossover:

            reverse_strategy = 1
            time_in_trade = 1
            buy_sequence = []
            sell_sequence = []
            
        elif reverse_strategy == 1 and ((time_in_trade > time_in_trade_limit )  or (sequence_alarm)): # or ((percentage_stoploss))
            

            reverse_strategy = 0
            time_in_trade = 1             
                

        if reverse_strategy == 0:
            
            if cond == 1 :
                
                if not sequence_alarm:
                    buy_points.append(np.nan)
                    sell_sequence.append(cross_df.iloc[i,]["Close"])
                    sell_points.append(cross_df.iloc[i,]["Close"])
                    num_of_trades += 1
                    cond = 0
                
                if first_changer == -1:
                    for i in range(0,len(sell_sequence) - 1):
                        wallet += fee * wallet * ((sell_sequence[i] - buy_sequence[i])/sell_sequence[i]) * leverage
                        if (sell_sequence[i] - buy_sequence[i] )> 0:
                            num_of_wins += 1
                            
                        wallet += fee * wallet * ((sell_sequence[i+1] - buy_sequence[i])/buy_sequence[i]) * leverage
                        if (sell_sequence[i+1] - buy_sequence[i]) > 0:
                            num_of_wins += 1
                            
                elif first_changer == 1:
                    for i in range(0,len(buy_sequence)-1):
                        wallet += fee * wallet * ((sell_sequence[i] - buy_sequence[i])/ buy_sequence[i]) * leverage
                        if (sell_sequence[i] - buy_sequence[i] )> 0:
                            num_of_wins += 1
                            
                        wallet += fee * wallet * ((sell_sequence[i] - buy_sequence[i])/sell_sequence[i]) * leverage
                        if (sell_sequence[i] - buy_sequence[i]) > 0:
                            num_of_wins += 1
                            
                    wallet += fee * wallet * ((sell_sequence[-1] - buy_sequence[-1])/buy_sequence[-1]) * leverage
                    if (sell_sequence[-1] - buy_sequence[-1]) > 0:
                        num_of_wins += 1
                        
            elif cond == -1:
                
                
                if not sequence_alarm:
                    buy_points.append(cross_df.iloc[i,]["Close"])
                    buy_sequence.append(cross_df.iloc[i,]["Close"])
                    num_of_trades += 1
                    sell_points.append(np.nan)
                    cond = 0
                
                if first_changer == 1:
                    for i in range(0,len(buy_sequence) -1 ):
                        wallet += fee * wallet * ((sell_sequence[i] - buy_sequence[i])/buy_sequence[i]) * leverage
                        if (sell_sequence[i] - buy_sequence[i] )> 0:
                            num_of_wins += 1
                        wallet += fee * wallet * ((sell_sequence[i] - buy_sequence[i + 1])/sell_sequence[i]) * leverage
                        if (sell_sequence[i] - buy_sequence[i+1] )> 0:
                            num_of_wins += 1
                            
                elif first_changer == -1:
                    for i in range(0, len(sell_sequence)-1):
                        wallet += fee * wallet * ((sell_sequence[i] - buy_sequence[i])/sell_sequence[i]) * leverage
                        if (sell_sequence[i] - buy_sequence[i] )> 0:
                            num_of_wins += 1
                        wallet += fee * wallet * ((sell_sequence[i + 1] - buy_sequence[i])/ buy_sequence[i]) * leverage
                        if (sell_sequence[i+1] - buy_sequence[i] )> 0:
                            num_of_wins += 1
                    wallet += fee * wallet * ((sell_sequence[-1] - buy_sequence[-1])/sell_sequence[-1]) * leverage
                    if (sell_sequence[-1] - buy_sequence[-1] )> 0:
                        num_of_wins += 1
                
            elif cond == 0 :
                buy_points.append(np.nan)
                sell_points.append(np.nan)
                
        elif reverse_strategy == 1:
            
            if cross_df.iloc[i,]["Close"]< cross_df.iloc[i,][index1]  and cond != 1  :  

                time_in_trade = 1
                if first_changer == 0:
                    first_changer = 1
                cond = 1
                buy_sequence.append(cross_df.iloc[i,]["Close"])
                buy_points.append(cross_df.iloc[i,]["Close"])
                num_of_trades += 1
        
                sell_points.append(np.nan)
                #sell_sequence.append(np.nan)

    
    
            elif (cross_df.iloc[i,]["Close"] > cross_df.iloc[i,][index1]) and cond != -1  :  
  
                time_in_trade = 1
                if first_changer == 0:
                    first_changer = -1
                cond = -1
                sell_sequence.append(cross_df.iloc[i,]["Close"])
                sell_points.append(cross_df.iloc[i,]["Close"])
                num_of_trades += 1

                #buy_sequence.append(np.nan)
                buy_points.append(np.nan)

            else:
                time_in_trade += 1
                buy_points.append(np.nan)
                sell_points.append(np.nan)
                
    hodling_wallet = 10000*(cross_df.iloc[-1,]["Close"])/cross_df.iloc[0,]["Close"] 
                
    if show_on_graph == True:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,15))
        plt.plot(cross_df["Close"], linewidth = 3, alpha = 0.2)
        plt.plot(cross_df[index1], linewidth = 3, alpha = 0.8)
        plt.grid()
        plt.scatter(cross_df.index, buy_points, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points, marker = '1', color = 'red')
        
    return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Win Probability" : num_of_wins/num_of_trades, "No of Trade" : num_of_trades}

print(trade(df_1h, end_time_1h - 300*day, end_time_1h - 100*day, 270, True, True, False, 0.99925, 2, 1))
#trade(orig_df, start_time, end_time, ema_period, long, short, show_on_graph, fee, time_in_trade_limit, leverage)
            