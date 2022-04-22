# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 23:00:12 2022

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

def trade(orig_df, start_time, end_time, ema_period, long, short, show_on_graph, fee, x, y, fisher_ema_period, time_in_trade_limit):
                                                                                     #ema p. # ma. #fisher ema per.  

    df = orig_df.copy(deep = True)
    index1 = "EMA" + str(ema_period)
    df[index1] = ema(df["Close"], ema_period)
    
    index2 = "EMA" + str(x)
    index3 = "SMA" + str(y)    
    df[index2] = ema(df["Close"], x)
    df[index3] = df["Close"].rolling(window = y).mean()   
    #rsi_ = rsi(df, start_time, end_time, 14, False)
    cross_df = pd.DataFrame()
    cross_df["Close"] = df.loc[start_time : end_time, "Close"]
    #cross_df["Fisher"] = df.loc[start_time : end_time, "Fisher"]
    #cross_df["RSI"] = rsi_
    cross_df[index1] = df.loc[start_time : end_time, index1]
    cross_df["Volume"] = df.loc[start_time : end_time, "Volume"]
    cross_df[index2] = df.loc[start_time : end_time, index2]
    cross_df[index3] = df.loc[start_time : end_time, index3]

    temporary_df = pd.DataFrame()
    temporary_df["Close"] = df.loc[(start_time - fisher_ema_period*day/24) : end_time, index2]
    temp_fish = pd.DataFrame()
    temp_fish["Fisher"] = (fisher_transform(temporary_df, fisher_ema_period, False))["Fisher"]
    cross_df[str(index2 + "Fisher")] = temp_fish.loc[start_time : end_time, "Fisher"]

    trades_df = pd.DataFrame()
    
    wallet = 10000
    buy_points = []
    sell_points = []
    buy_volumes = []
    sell_volumes = []
    cond = 0
    first_changer = 0
    time_in_trade = 1
    times_in_trades = []
    reverse_strategy = 0

    
    
    for i in range(0,len(cross_df)):

#double_ma_trade(df, x, y, start_time, end_time, mov_avg_type, long, short, show_on_graph, fee, ema_fisher_period): 
        crossover = ((cross_df.iloc[i-1,]["Close"]< cross_df.iloc[i-1,][index1]) and  (cross_df.iloc[i,]["Close"]> cross_df.iloc[i,][index1])) or ((cross_df.iloc[i-1,]["Close"]> cross_df.iloc[i-1,][index1]) and  (cross_df.iloc[i,]["Close"]< cross_df.iloc[i,][index1]))
        if reverse_strategy == 0 and crossover:
            
            reverse_strategy = 1
            time_in_trade = 1
        
        elif reverse_strategy == 1 and (time_in_trade > time_in_trade_limit ): # or (extreme_volume) or ((percentage_stoploss))
            
            reverse_strategy = 0
            time_in_trade = 1             
                

        if reverse_strategy == 0:
            if cross_df.iloc[i,][index2]-cross_df.iloc[i,][index3] > 0 and cond != 1 and not cross_df.iloc[i,][str(index2 + "Fisher")] < -2: 
                if first_changer != 0:
                    times_in_trades.append(time_in_trade)
                    time_in_trade = 1       
                
                if first_changer == 0:
                    first_changer = 1
                cond = 1
                buy_points.append(cross_df.iloc[i,]["Close"])
                buy_volumes.append(cross_df.iloc[i,]["Volume"])
                sell_points.append(np.nan)
                sell_volumes.append(np.nan)
            elif (cross_df.iloc[i,][index2] - cross_df.iloc[i,][index3]) < 0 and cond != -1     and not (cross_df.iloc[i,][str(index2 + "Fisher")] > 2):
                
                if first_changer != 0:
                    times_in_trades.append(time_in_trade)   
                    time_in_trade = 1    
                if first_changer == 0:
                    first_changer = -1
                cond = -1
                sell_points.append(cross_df.iloc[i,]["Close"])
                sell_volumes.append(cross_df.iloc[i,]["Volume"])
                buy_points.append(np.nan)
                buy_volumes.append(np.nan)
            else:
                time_in_trade += 1
                buy_points.append(np.nan)
                sell_points.append(np.nan)
                buy_volumes.append(np.nan)
                sell_volumes.append(np.nan)  

        elif reverse_strategy == 1:
            if cross_df.iloc[i,]["Close"]< cross_df.iloc[i,][index1]  and cond != 1  :  
                if first_changer != 0:
                    times_in_trades.append(time_in_trade)
                    time_in_trade = 1
                if first_changer == 0:
                    first_changer = 1
                cond = 1
                buy_points.append(cross_df.iloc[i,]["Close"])
                buy_volumes.append(cross_df.iloc[i,]["Volume"])
                sell_points.append(np.nan)
                sell_volumes.append(np.nan)
    
    
            elif (cross_df.iloc[i,]["Close"] > cross_df.iloc[i,][index1]) and cond != -1  :  
                if first_changer != 0:
                    times_in_trades.append(time_in_trade)   
                    time_in_trade = 1
                if first_changer == 0:
                    first_changer = -1
                cond = -1
                sell_points.append(cross_df.iloc[i,]["Close"])
                sell_volumes.append(cross_df.iloc[i,]["Volume"])
                buy_points.append(np.nan)
                buy_volumes.append(np.nan)
            else:
                time_in_trade += 1
                buy_points.append(np.nan)
                sell_points.append(np.nan)
                buy_volumes.append(np.nan)
                sell_volumes.append(np.nan)  
            
            
 
   
    trades_df["Buy Volumes"] = buy_volumes
    trades_df["Sell Volumes"] = sell_volumes
    trades_df["Buy Points"] = buy_points
    trades_df["Sell Points"] = sell_points
    trades_df["Win Situation"] = np.nan
    trades_df["Return on Trade"] = np.nan
    trades_df["Max Drawdown"] = np.nan
    trades_df["Max Drawup"] = np.nan


    
    geniune_sell_points = list(pd.Series(sell_points).dropna())
    geniune_buy_points = list(pd.Series(buy_points).dropna())

    sell_points = pd.Series(sell_points).fillna(0)
    buy_points= pd.Series(buy_points).fillna(0)
        
    if first_changer == 1:

        if len(geniune_buy_points) > len(geniune_sell_points):
            geniune_sell_points.append(cross_df.iloc[-1,]["Close"])
            
#        if len(geniune_buy_points) == len(geniune_sell_points):
#            geniune_sell_points.append(cross_df.iloc[-1,]["Close"])            
     
            
        for k in range(0,len(geniune_sell_points)-1):
            if long == True:
                wallet += wallet*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_buy_points[k]
                wallet *= fee
            if short == True:
                wallet += wallet*(geniune_sell_points[k]-geniune_buy_points[k+1])/geniune_sell_points[k]
                wallet *= fee
                
        if long==True:
            wallet += wallet*(geniune_sell_points[-1]-geniune_buy_points[-1])/geniune_buy_points[-1]
            wallet *= fee
            
    elif first_changer == -1:

        if len(geniune_buy_points) == len(geniune_sell_points) :
            geniune_sell_points.append(cross_df.iloc[-1,]["Close"])
            for k in range(0,len(geniune_buy_points)):
                if short == True:
                    wallet += wallet*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_sell_points[k]
                    wallet *= fee
                if long == True:
                    wallet += wallet*(geniune_sell_points[k+1]-geniune_buy_points[k])/geniune_buy_points[k]
                    wallet *= fee


        elif len(geniune_sell_points) > len(geniune_buy_points):
            geniune_buy_points.append(cross_df.iloc[-1,]["Close"])
            for k in range(0,len(geniune_sell_points)-1):
                if short == True:
                    wallet += wallet*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_sell_points[k]
                    wallet *= fee
                if long == True:
                    wallet +=  wallet*((geniune_sell_points[k+1]-geniune_buy_points[k])/geniune_buy_points[k])
                    wallet *= fee
                    
            if short == True:
                wallet += wallet*(geniune_sell_points[-1]-geniune_buy_points[-1])/geniune_sell_points[-1]
                wallet *= fee

    buy_points = buy_points.replace(0, np.nan)
    sell_points = sell_points.replace(0, np.nan)

    
    wins =0
    losses= 0

    if len(geniune_buy_points) == len(geniune_sell_points) and first_changer == 1:
        trade_open_close_long = [[geniune_buy_points[i], geniune_sell_points[i] ] for i in range(0,len(geniune_buy_points)) ]
        trade_open_close_short = [[geniune_sell_points[i], geniune_buy_points[i+1] ] for i in range(0,len(geniune_buy_points)-1)]
        if long == True:
            for every in trade_open_close_long:
                if every[0] < every[1]:
                    wins += 1
                elif every[0] > every[1]:
                    losses += 1
        if short == True:
            for every in trade_open_close_short:
                if every[0] > every[1]:
                    wins += 1
                elif every[0] < every[1]:
                    losses += 1       
            
    elif len(geniune_buy_points) == len(geniune_sell_points) and first_changer == -1:
        trade_open_close_long = [[geniune_buy_points[i], geniune_sell_points[i+1] ] for i in range(0,len(geniune_buy_points)-1) ]
        trade_open_close_short = [[geniune_sell_points[i], geniune_buy_points[i] ] for i in range(0,len(geniune_buy_points)) ]
        if long == True:
            for every in trade_open_close_long:
                if every[0] < every[1]:
                    wins += 1
                elif every[0] > every[1]:
                    losses += 1
        if short == True:
            for every in trade_open_close_short:
                if every[0] > every[1]:
                    wins += 1
                elif every[0] < every[1]:
                    losses += 1    
    elif len(geniune_sell_points) > len(geniune_buy_points) and first_changer == -1:
        trade_open_close_long = [[geniune_buy_points[i], geniune_sell_points[i+1] ] for i in range(0,len(geniune_buy_points)) ]
        trade_open_close_short = [[geniune_sell_points[i], geniune_buy_points[i] ] for i in range(0,len(geniune_sell_points)-1) ]
        if long == True:
            for every in trade_open_close_long:
                if every[0] < every[1]:
                    wins += 1
                elif every[0] > every[1]:
                    losses += 1
        if short == True:
            for every in trade_open_close_short:
                if every[0] > every[1]:
                    wins += 1
                elif every[0] < every[1]:
                    losses += 1        
        
    try:
        win_prob = wins/(wins+losses)
    except:
        win_prob= 0
        
        
    total_return = ((wallet)*100/10000) - 100
    hodling_wallet = 10000*(cross_df.iloc[-1,]["Close"])/cross_df.iloc[0,]["Close"] 
    
    if long == True and short == True:
        num_of_trade = len(trade_open_close_long)+len(trade_open_close_short)
    elif long == True:
        num_of_trade = len(trade_open_close_long)
    elif short == True:
        num_of_trade = len(trade_open_close_short)
        
    

    long_trade_performances = [100*(every[1]-every[0])/every[0]  for every in trade_open_close_long]
    short_trade_performances =  [100*(every[0]-every[1])/every[0]  for every in trade_open_close_short]
    all_returns = long_trade_performances +  short_trade_performances
    
    mean_of_winning_trades = (sum([i for i in all_returns if i >0]) / len([i for i in all_returns if i >0]))
    mean_of_losing_trades = (sum([i for i in all_returns if i <0]) / len([i for i in all_returns if i <0]))
    ratio_win_to_lose_return_mean = mean_of_winning_trades*wins/(mean_of_losing_trades * losses)
    
    long_return_mean = sum(long_trade_performances)/len(long_trade_performances) 
    short_return_mean = sum(short_trade_performances)/len(short_trade_performances) 
    total_trades_return_mean = (sum(all_returns))/ (len(all_returns) )
   
    std_of_total_trades_return_mean = sum([((x - total_trades_return_mean) ** 2) for x in all_returns]) / (len(all_returns)**(0.5))
    half_length = ((t.ppf( 0.90 + ((1-0.90)/2), len(all_returns) - 1 ))*std_of_total_trades_return_mean) / (len(all_returns)**(0.5))
    conf_interval = (total_trades_return_mean + half_length , total_trades_return_mean - half_length)
    
    
    max_returns_for_long_trades =[]
    min_returns_for_long_trades = []
    for every in trade_open_close_long:
        start = (cross_df[cross_df["Close"] == every[0]].index.values)[0]
        end = (cross_df[cross_df["Close"] == every[1]].index.values)[0]
        
        idx_1 = trades_df.index[trades_df['Buy Points'] == every[0]]
        
        if every[1] > every[0]:
            trades_df["Win Situation"][idx_1] = 1
            trades_df["Return on Trade"][idx_1] = 0.99925*(every[1]-every[0])*100/every[0]
            
        elif every[1] < every[0]:
            trades_df["Win Situation"][idx_1] = -1      
            trades_df["Return on Trade"][idx_1] = 1.00075*(every[1]-every[0])*100/every[0]
        
    
        try:
            max_value = max(cross_df.loc[start : end ,"Close"])
            min_value = min(cross_df.loc[start : end ,"Close"])

            max_return = 100 * (max_value - every[0] )/every[0]
            min_return = 100 * (min_value - every[0] )/every[0]

            max_returns_for_long_trades.append(max_return)
            min_returns_for_long_trades.append(min_return)
            
        except:
            start = (cross_df[cross_df["Close"] == every[0]].index.values)[0]
            end = (cross_df[cross_df["Close"] == every[1]].index.values)[1]
            
            max_value = max(cross_df.loc[start : end ,"Close"])
            min_value = min(cross_df.loc[start : end ,"Close"])

            max_return = 100 * (every[0] - min_value)/every[0]
            min_return = 100 * (every[0] - max_value)/every[0]
            
            max_returns_for_long_trades.append(max_return)
            min_returns_for_long_trades.append(min_return)
            
        trades_df["Max Drawdown"][idx_1] =  min_return
        trades_df["Max Drawup"][idx_1] =  max_return    
    
#    zipped_returns_long = zip(long_trade_performances , max_returns_for_long_trades, min_returns_for_long_trades)
    
    
    max_returns_for_short_trades =[]
    min_returns_for_short_trades = []
    for every in trade_open_close_short:
        start = (cross_df[cross_df["Close"] == every[0]].index.values)[0]
        end = (cross_df[cross_df["Close"] == every[1]].index.values)[0]
        

        idx_1 = trades_df.index[trades_df['Sell Points'] == every[0]]
        
        if every[1] > every[0]:
            trades_df["Win Situation"][idx_1] = -1
            trades_df["Return on Trade"][idx_1] = 1.00075*(every[0]-every[1])*100/every[0]
        elif every[1] < every[0]:
            trades_df["Win Situation"][idx_1] = +1   
            trades_df["Return on Trade"][idx_1] = 0.99925*(every[0]-every[1])*100/every[0]
            
        try:
            max_value = max(cross_df.loc[start : end ,"Close"])
            min_value = min(cross_df.loc[start : end ,"Close"])

            max_return = 100 * (every[0] - min_value)/every[0]
            min_return = 100 * (every[0] - max_value)/every[0]

            max_returns_for_short_trades.append(max_return)
            min_returns_for_short_trades.append(min_return)
            
        except:
            start = (cross_df[cross_df["Close"] == every[0]].index.values)[0]
            end = (cross_df[cross_df["Close"] == every[1]].index.values)[1]
            
            max_value = max(cross_df.loc[start : end ,"Close"])
            min_value = min(cross_df.loc[start : end ,"Close"])

            max_return = 100 * (every[0] - min_value)/every[0]
            min_return = 100 * (every[0] - max_value)/every[0]
            
            max_returns_for_short_trades.append(max_return)
            min_returns_for_short_trades.append(min_return)
            
        trades_df["Max Drawdown"][idx_1] =  min_return
        trades_df["Max Drawup"][idx_1] =  max_return 
        
    trades_df = trades_df[trades_df['Win Situation'].notna()]

    try:
        trades_df["Time in Trade"]= times_in_trades
    except:
        times_in_trades.append(np.average(times_in_trades))
        trades_df["Time in Trade"]= times_in_trades
        
    if show_on_graph == True:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,15))
        plt.plot(cross_df["Close"], linewidth = 3, alpha = 0.2)
        plt.plot(cross_df[index1], linewidth = 3, alpha = 0.8)
        plt.plot(cross_df[index2], linewidth = 3, alpha = 0.8)
        plt.plot(cross_df[index3], linewidth = 3, alpha = 0.8)
        plt.grid()
        plt.scatter(cross_df.index, buy_points, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points, marker = '1', color = 'red')
        
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (20,20))

        plt.scatter(long_trade_performances,long_trade_performances,  marker = 'o',  color = 'green')
        plt.scatter(short_trade_performances,short_trade_performances, marker = 'o',  color = 'red')
        
        plt.scatter(long_trade_performances, max_returns_for_long_trades,  marker = 'o',  color = 'green', facecolors = 'none')
        plt.scatter(long_trade_performances, min_returns_for_long_trades,  marker = 'o',  color = 'green', facecolors = 'none')
        
        plt.scatter(short_trade_performances, max_returns_for_short_trades,  marker = 'o',  color = 'red', facecolors = 'none')
        plt.scatter(short_trade_performances, min_returns_for_short_trades,  marker = 'o',  color = 'red', facecolors = 'none')
        for i in range(0,len(long_trade_performances)):
            plt.plot([long_trade_performances[i], long_trade_performances[i]], [max_returns_for_long_trades[i], min_returns_for_long_trades[i]] , 'k--', linewidth = 1)
        for i in range(0,len(short_trade_performances)):
            plt.plot([short_trade_performances[i], short_trade_performances[i]], [max_returns_for_short_trades[i], min_returns_for_short_trades[i]], 'k--', linewidth = 1 )

        plt.grid()
    
    if long == True and short == False:
        return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return, "Long Trades": trade_open_close_long, 
                "Win Probability" : win_prob , "No of Trade" : num_of_trade}, trades_df
    
    if long == False and short == True:
        return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return,  "Short Trades": trade_open_close_short  ,
                "Win Probability" : win_prob , "No of Trade" : num_of_trade}, trades_df

    else:
        return [{"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return, 
                "Long Trades": trade_open_close_long, "Short Trades": trade_open_close_short  , 
                "Win Probability" : win_prob , "No of Trade" : num_of_trade, "Trades' Return Mean" : total_trades_return_mean,
                "%90 Percent T Distribution Confidence Interval for Trades' Return Mean" : conf_interval,
               "Long Trades' Return Mean" : long_return_mean, "Short Trades' Return Mean" : short_return_mean,
               "Winning Trade's Return Mean" : mean_of_winning_trades, "Losing Trade's Return Mean" : mean_of_losing_trades,
               "Weighted Ratio of Winning Trades Return Mean to Losing Trades Return Mean" : abs(ratio_win_to_lose_return_mean)}, trades_df]



def Trade_Analysis():
    pd.set_option('display.max_columns', None)
    
    trades_df = (trade(df_1h, end_time_1h - 200*day, end_time_1h ,270, True, True, False, 0.99925, 20, 20, 2, 9999)[1])
    # print(trades_df)
    
    
    # sell_trades = trades_df.copy(deep = True)
    # sell_trades= sell_trades[sell_trades["Sell Points"].notna()]
    
    # winning_short_trades = sell_trades[sell_trades["Win Situation"] == 1]
    # losing_short_trades = sell_trades[sell_trades["Win Situation"] == -1]
    # print(winning_short_trades[["Sell Volumes", "Sell Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup"]])
    # print(losing_short_trades[["Sell Volumes", "Sell Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup"]])
   
    
    # buy_trades = trades_df.copy(deep = True)
    # buy_trades= buy_trades[buy_trades["Buy Points"].notna()]
    
    # winning_long_trades = buy_trades[buy_trades["Win Situation"] == 1]
    # losing_long_trades = buy_trades[buy_trades["Win Situation"] == -1]
    # print(winning_long_trades[["Buy Volumes", "Buy Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup"]])
    # print(losing_long_trades[["Buy Volumes", "Buy Points", "Win Situation", "Return on Trade", "Time in Trade","Max Drawdown", "Max Drawup"]])
    
    
    trade_no_ind = list(range(0,len(trades_df)))
    trades_df['Trade No'] = trade_no_ind
    
    loss_df = trades_df.copy(deep = True)
    loss_df = loss_df[loss_df['Win Situation'] == -1]
    print(loss_df)
    times_till_next_loss = [(loss_df.iloc[i+1,]['Trade No'] - loss_df.iloc[i,]['Trade No']) for i in range(0,len(loss_df)-1)]
    plt.hist(times_till_next_loss, bins=np.arange(min(times_till_next_loss), max(times_till_next_loss)+1))
    plt.show()
    print(times_till_next_loss)
    
Trade_Analysis()
# result = (trade(df_1h, end_time_1h-100*day, end_time_1h- 0*day , 270, True, True, True, 0.99925, 20, 20, 50, 2)[0])
# keys = ["Wallet", "Hodling Wallet", "Win Probability", "No of Trade"]
# temp_result = {your_key: result[your_key] for your_key in keys}
#print(temp_result)
#trade(orig_df, start_time, end_time, ema_period, long, short, show_on_graph, fee, x, y, fisher_ema_period)


# def optimize_strategy():
#     for fisher_x in range(290,311,2):
#         for trade_time_limit in range(40,41):
#             result = (trade(df_1h, end_time_1h-300*day, end_time_1h- 100*day, 270, True, True, False, 0.99925, 20, 20, fisher_x, trade_time_limit)[0])
#             keys = ["Wallet", "Hodling Wallet", "Win Probability", "No of Trade"]
#             temp_result = {your_key: result[your_key] for your_key in keys}
#             print( fisher_x, trade_time_limit, temp_result)
# optimize_strategy()
"""
10 10 {'Wallet': 9499.112434266368, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5409429280397022, 'No of Trade': 403}
10 20 {'Wallet': 22196.879723107988, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6202898550724638, 'No of Trade': 345}
10 30 {'Wallet': 25662.240421156308, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6525679758308157, 'No of Trade': 331}
10 40 {'Wallet': 28452.032293501295, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6728971962616822, 'No of Trade': 321}
10 50 {'Wallet': 23854.675374178038, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6741214057507987, 'No of Trade': 313}
10 60 {'Wallet': 22701.57399392312, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6885245901639344, 'No of Trade': 305}
10 70 {'Wallet': 21762.482151726166, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6843853820598007, 'No of Trade': 301}
10 80 {'Wallet': 22864.72131110126, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.697594501718213, 'No of Trade': 291}
10 90 {'Wallet': 25103.440746535478, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7087719298245614, 'No of Trade': 285}
10 100 {'Wallet': 30014.346256056655, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7240143369175627, 'No of Trade': 279}
10 110 {'Wallet': 31168.278663593144, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7326007326007326, 'No of Trade': 273}
10 120 {'Wallet': 29185.709208946904, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7360594795539034, 'No of Trade': 269}
20 10 {'Wallet': 6024.614305316096, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5386533665835411, 'No of Trade': 401}
20 20 {'Wallet': 13930.841509227892, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6080691642651297, 'No of Trade': 347}
20 30 {'Wallet': 16370.374979605847, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6396396396396397, 'No of Trade': 333}
20 40 {'Wallet': 18137.230619175585, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6563467492260062, 'No of Trade': 323}
20 50 {'Wallet': 15611.21631328438, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6571428571428571, 'No of Trade': 315}
20 60 {'Wallet': 15976.126310301835, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6775244299674267, 'No of Trade': 307}
20 70 {'Wallet': 15893.79356542621, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6765676567656765, 'No of Trade': 303}
20 80 {'Wallet': 16788.36954772, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6907216494845361, 'No of Trade': 291}
20 90 {'Wallet': 18561.89914565494, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7017543859649122, 'No of Trade': 285}
20 100 {'Wallet': 22609.580733222196, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7168458781362007, 'No of Trade': 279}
20 110 {'Wallet': 23573.040636775495, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7216117216117216, 'No of Trade': 273}
20 120 {'Wallet': 21421.252253627288, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.724907063197026, 'No of Trade': 269}
30 10 {'Wallet': 4772.0099600691865, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.531328320802005, 'No of Trade': 399}
30 20 {'Wallet': 12165.025346490547, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6051873198847262, 'No of Trade': 347}
30 30 {'Wallet': 14762.74831610686, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6396396396396397, 'No of Trade': 333}
30 40 {'Wallet': 16301.80483274096, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6563467492260062, 'No of Trade': 323}
30 50 {'Wallet': 14078.141620361841, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6571428571428571, 'No of Trade': 315}
30 60 {'Wallet': 14884.260754946263, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6775244299674267, 'No of Trade': 307}
30 70 {'Wallet': 14807.554924033477, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6765676567656765, 'No of Trade': 303}
30 80 {'Wallet': 15640.992387343136, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6907216494845361, 'No of Trade': 291}
30 90 {'Wallet': 17293.312635666232, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7017543859649122, 'No of Trade': 285}
30 100 {'Wallet': 21064.36120101822, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7168458781362007, 'No of Trade': 279}
30 110 {'Wallet': 21961.9748122836, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7216117216117216, 'No of Trade': 273}
30 120 {'Wallet': 19957.247335661858, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.724907063197026, 'No of Trade': 269}
40 10 {'Wallet': 5774.468335037371, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5340050377833753, 'No of Trade': 397}
40 20 {'Wallet': 13246.05226531832, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6051873198847262, 'No of Trade': 347}
40 30 {'Wallet': 16105.155834830297, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6396396396396397, 'No of Trade': 333}
40 40 {'Wallet': 17784.161973000355, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6563467492260062, 'No of Trade': 323}
40 50 {'Wallet': 15358.296423259064, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6571428571428571, 'No of Trade': 315}
40 60 {'Wallet': 15898.127395421214, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6775244299674267, 'No of Trade': 307}
40 70 {'Wallet': 15816.196616869247, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6765676567656765, 'No of Trade': 303}
40 80 {'Wallet': 16718.773220843745, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6907216494845361, 'No of Trade': 291}
40 90 {'Wallet': 18478.494405639973, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7017543859649122, 'No of Trade': 285}
40 100 {'Wallet': 21975.000650700644, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7168458781362007, 'No of Trade': 279}
40 110 {'Wallet': 22911.419253828353, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7216117216117216, 'No of Trade': 273}
40 120 {'Wallet': 20820.0248278198, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.724907063197026, 'No of Trade': 269}
50 10 {'Wallet': 5841.65891372246, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5345268542199488, 'No of Trade': 391}
50 20 {'Wallet': 15338.264540663446, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6151603498542274, 'No of Trade': 343}
50 30 {'Wallet': 18280.86186961997, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6504559270516718, 'No of Trade': 329}
50 40 {'Wallet': 20535.03066958059, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.664576802507837, 'No of Trade': 319}
50 50 {'Wallet': 18363.364288792796, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6677316293929713, 'No of Trade': 313}
50 60 {'Wallet': 19259.3449524139, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6885245901639344, 'No of Trade': 305}
50 70 {'Wallet': 18700.722515990477, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6877076411960132, 'No of Trade': 301}
50 80 {'Wallet': 19836.197162747496, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7024221453287197, 'No of Trade': 289}
50 90 {'Wallet': 21924.040326357368, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7137809187279152, 'No of Trade': 283}
50 100 {'Wallet': 25800.828685611144, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7292418772563177, 'No of Trade': 277}
50 110 {'Wallet': 26900.27693325189, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7343173431734318, 'No of Trade': 271}
50 120 {'Wallet': 24444.77260097932, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7378277153558053, 'No of Trade': 267}
60 10 {'Wallet': 6051.456087524078, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5372750642673522, 'No of Trade': 389}
60 20 {'Wallet': 14929.312929781956, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6158357771260997, 'No of Trade': 341}
60 30 {'Wallet': 17788.61945547516, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6513761467889908, 'No of Trade': 327}
60 40 {'Wallet': 19968.598338159885, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6656151419558359, 'No of Trade': 317}
60 50 {'Wallet': 18281.889790724974, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6688102893890675, 'No of Trade': 311}
60 60 {'Wallet': 19105.710698992105, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6897689768976898, 'No of Trade': 303}
60 70 {'Wallet': 18791.18476585081, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6889632107023411, 'No of Trade': 299}
60 80 {'Wallet': 19932.152119700706, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7038327526132404, 'No of Trade': 287}
60 90 {'Wallet': 22030.094946025427, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7153024911032029, 'No of Trade': 281}
60 100 {'Wallet': 26436.134398776067, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.730909090909091, 'No of Trade': 275}
60 110 {'Wallet': 27562.654868071666, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7360594795539034, 'No of Trade': 269}
60 120 {'Wallet': 25046.687519281146, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7396226415094339, 'No of Trade': 265}
70 10 {'Wallet': 6213.350257234198, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5319693094629157, 'No of Trade': 391}
70 20 {'Wallet': 15844.418259393728, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.616519174041298, 'No of Trade': 339}
70 30 {'Wallet': 18878.988486300954, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6523076923076923, 'No of Trade': 325}
70 40 {'Wallet': 21192.591086524986, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6666666666666666, 'No of Trade': 315}
70 50 {'Wallet': 19402.786809031135, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6731391585760518, 'No of Trade': 309}
70 60 {'Wallet': 20274.692072084064, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6910299003322259, 'No of Trade': 301}
70 70 {'Wallet': 19799.396470256324, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6902356902356902, 'No of Trade': 297}
70 80 {'Wallet': 21051.04490389561, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7052631578947368, 'No of Trade': 285}
70 90 {'Wallet': 22643.460608530444, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7168458781362007, 'No of Trade': 279}
70 100 {'Wallet': 27171.938877967557, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7326007326007326, 'No of Trade': 273}
70 110 {'Wallet': 28331.96061552065, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7378277153558053, 'No of Trade': 267}
70 120 {'Wallet': 25686.7514133447, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7414448669201521, 'No of Trade': 263}
80 10 {'Wallet': 6496.328928789847, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5370843989769821, 'No of Trade': 391}
80 20 {'Wallet': 16617.39794554431, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6224188790560472, 'No of Trade': 339}
80 30 {'Wallet': 19887.717634832752, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6584615384615384, 'No of Trade': 325}
80 40 {'Wallet': 22786.09035641901, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6730158730158731, 'No of Trade': 315}
80 50 {'Wallet': 20861.708301352013, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6796116504854369, 'No of Trade': 309}
80 60 {'Wallet': 21806.449206344012, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6976744186046512, 'No of Trade': 301}
80 70 {'Wallet': 21298.414214538447, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.696969696969697, 'No of Trade': 297}
80 80 {'Wallet': 22467.397550923746, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.712280701754386, 'No of Trade': 285}
80 90 {'Wallet': 24508.58295899918, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7240143369175627, 'No of Trade': 279}
80 100 {'Wallet': 29424.32636731243, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.73992673992674, 'No of Trade': 273}
80 110 {'Wallet': 30776.90607385519, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7490636704119851, 'No of Trade': 267}
80 120 {'Wallet': 27903.424910095968, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7490494296577946, 'No of Trade': 263}
90 10 {'Wallet': 8312.29863510731, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5347043701799485, 'No of Trade': 389}
90 20 {'Wallet': 19502.512234515965, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6201780415430267, 'No of Trade': 337}
90 30 {'Wallet': 23340.62515448899, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6563467492260062, 'No of Trade': 323}
90 40 {'Wallet': 28033.575804211716, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.670926517571885, 'No of Trade': 313}
90 50 {'Wallet': 24572.272142889702, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6775244299674267, 'No of Trade': 307}
90 60 {'Wallet': 25685.04921208492, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6956521739130435, 'No of Trade': 299}
90 70 {'Wallet': 25087.635444218864, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6983050847457627, 'No of Trade': 295}
90 80 {'Wallet': 26579.436728822027, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7137809187279152, 'No of Trade': 283}
90 90 {'Wallet': 23669.74527982314, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7168458781362007, 'No of Trade': 279}
90 100 {'Wallet': 28417.24106651941, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7326007326007326, 'No of Trade': 273}
90 110 {'Wallet': 29723.52699819011, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7415730337078652, 'No of Trade': 267}
90 120 {'Wallet': 26948.394411931105, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7414448669201521, 'No of Trade': 263}
100 10 {'Wallet': 8726.978314452008, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5326370757180157, 'No of Trade': 383}
100 20 {'Wallet': 20068.276921648077, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.622356495468278, 'No of Trade': 331}
100 30 {'Wallet': 23599.572744335415, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6593059936908517, 'No of Trade': 317}
100 40 {'Wallet': 28344.588334562966, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6742671009771987, 'No of Trade': 307}
100 50 {'Wallet': 24844.88397767721, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6810631229235881, 'No of Trade': 301}
100 60 {'Wallet': 25970.006514836412, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6996587030716723, 'No of Trade': 293}
100 70 {'Wallet': 25365.964867283914, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7024221453287197, 'No of Trade': 289}
100 80 {'Wallet': 26874.316623205752, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7184115523465704, 'No of Trade': 277}
100 90 {'Wallet': 23932.344222735832, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7216117216117216, 'No of Trade': 273}
100 100 {'Wallet': 28952.719661937554, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7415730337078652, 'No of Trade': 267}
100 110 {'Wallet': 30669.816921970752, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7471264367816092, 'No of Trade': 261}
100 120 {'Wallet': 27806.334120621443, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7509727626459144, 'No of Trade': 257}
110 10 {'Wallet': 8559.186006526621, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5271317829457365, 'No of Trade': 387}
110 20 {'Wallet': 20165.496653422742, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.622356495468278, 'No of Trade': 331}
110 30 {'Wallet': 23713.899656464342, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6593059936908517, 'No of Trade': 317}
110 40 {'Wallet': 28481.902229817053, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6742671009771987, 'No of Trade': 307}
110 50 {'Wallet': 24965.243735802538, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6810631229235881, 'No of Trade': 301}
110 60 {'Wallet': 26188.947638308586, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6996587030716723, 'No of Trade': 293}
110 70 {'Wallet': 25579.813594769796, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7024221453287197, 'No of Trade': 289}
110 80 {'Wallet': 27100.881567294953, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7184115523465704, 'No of Trade': 277}
110 90 {'Wallet': 24134.106757083115, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7216117216117216, 'No of Trade': 273}
110 100 {'Wallet': 29192.16093348918, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7415730337078652, 'No of Trade': 267}
110 110 {'Wallet': 30541.008893006267, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7471264367816092, 'No of Trade': 261}
110 120 {'Wallet': 27689.5522337286, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7509727626459144, 'No of Trade': 257}
120 10 {'Wallet': 8409.254223483387, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5324675324675324, 'No of Trade': 385}
120 20 {'Wallet': 19812.256419258705, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6291793313069909, 'No of Trade': 329}
120 30 {'Wallet': 23298.50183058581, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6666666666666666, 'No of Trade': 315}
120 40 {'Wallet': 27982.98301220422, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6819672131147541, 'No of Trade': 305}
120 50 {'Wallet': 24527.92603940446, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6889632107023411, 'No of Trade': 299}
120 60 {'Wallet': 25730.194246053452, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7079037800687286, 'No of Trade': 291}
120 70 {'Wallet': 25131.730440688098, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7142857142857143, 'No of Trade': 287}
120 80 {'Wallet': 26205.09310238528, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7220216606498195, 'No of Trade': 277}
120 90 {'Wallet': 23336.381620717613, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7252747252747253, 'No of Trade': 273}
120 100 {'Wallet': 28230.496183432515, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7453183520599251, 'No of Trade': 267}
120 110 {'Wallet': 29802.38214544938, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7509578544061303, 'No of Trade': 261}
120 120 {'Wallet': 27019.887260336363, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.754863813229572, 'No of Trade': 257}
130 10 {'Wallet': 11454.262529160818, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5404699738903395, 'No of Trade': 383}
130 20 {'Wallet': 23184.012721916388, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6299694189602446, 'No of Trade': 327}
130 30 {'Wallet': 27263.5661184374, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6677316293929713, 'No of Trade': 313}
130 40 {'Wallet': 32764.09272606217, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6831683168316832, 'No of Trade': 303}
130 50 {'Wallet': 28718.712468308036, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6902356902356902, 'No of Trade': 297}
130 60 {'Wallet': 30126.397524153042, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7093425605536332, 'No of Trade': 289}
130 70 {'Wallet': 29425.681535309683, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7157894736842105, 'No of Trade': 285}
130 80 {'Wallet': 30682.436533917273, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7236363636363636, 'No of Trade': 275}
130 90 {'Wallet': 27323.583442783838, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7269372693726938, 'No of Trade': 271}
130 100 {'Wallet': 33055.49546293929, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7471698113207547, 'No of Trade': 265}
130 110 {'Wallet': 34881.23536717288, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752895752895753, 'No of Trade': 259}
130 120 {'Wallet': 32058.5584466518, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7568627450980392, 'No of Trade': 255}
140 10 {'Wallet': 11095.204118967033, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5426356589147286, 'No of Trade': 387}
140 20 {'Wallet': 21312.05613700949, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6269113149847095, 'No of Trade': 327}
140 30 {'Wallet': 25062.21242114546, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6645367412140575, 'No of Trade': 313}
140 40 {'Wallet': 30126.835344974686, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6798679867986799, 'No of Trade': 303}
140 50 {'Wallet': 26413.895313457582, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6868686868686869, 'No of Trade': 297}
140 60 {'Wallet': 27175.53393063908, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7058823529411765, 'No of Trade': 289}
140 70 {'Wallet': 26507.223788337138, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7157894736842105, 'No of Trade': 285}
140 80 {'Wallet': 29025.150020290203, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.72, 'No of Trade': 275}
140 90 {'Wallet': 25326.301424708832, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7232472324723247, 'No of Trade': 271}
140 100 {'Wallet': 30639.225765923256, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7433962264150943, 'No of Trade': 265}
140 110 {'Wallet': 32331.508889568533, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.749034749034749, 'No of Trade': 259}
140 120 {'Wallet': 29707.304260297857, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7490196078431373, 'No of Trade': 255}
150 10 {'Wallet': 12039.20820268842, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5452196382428941, 'No of Trade': 387}
150 20 {'Wallet': 23125.33219846061, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6299694189602446, 'No of Trade': 327}
150 30 {'Wallet': 27194.559930842064, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6677316293929713, 'No of Trade': 313}
150 40 {'Wallet': 32690.091981834827, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6831683168316832, 'No of Trade': 303}
150 50 {'Wallet': 28661.24694174078, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6902356902356902, 'No of Trade': 297}
150 60 {'Wallet': 29487.687428020792, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7093425605536332, 'No of Trade': 289}
150 70 {'Wallet': 28762.516006128022, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7157894736842105, 'No of Trade': 285}
150 80 {'Wallet': 31494.67287502897, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7236363636363636, 'No of Trade': 275}
150 90 {'Wallet': 27481.11820087018, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7269372693726938, 'No of Trade': 271}
150 100 {'Wallet': 33246.07769356378, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7471698113207547, 'No of Trade': 265}
150 110 {'Wallet': 35299.58317916421, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752895752895753, 'No of Trade': 259}
150 120 {'Wallet': 32434.47317435476, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7529411764705882, 'No of Trade': 255}
160 10 {'Wallet': 14587.661558798027, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5478036175710594, 'No of Trade': 387}
160 20 {'Wallet': 28633.322040579616, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6330275229357798, 'No of Trade': 327}
160 30 {'Wallet': 34349.76868623531, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.670926517571885, 'No of Trade': 313}
160 40 {'Wallet': 40476.21557256384, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6897689768976898, 'No of Trade': 303}
160 50 {'Wallet': 34272.928305890346, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6949152542372882, 'No of Trade': 295}
160 60 {'Wallet': 35261.180338083286, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7142857142857143, 'No of Trade': 287}
160 70 {'Wallet': 34394.025178974895, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7208480565371025, 'No of Trade': 283}
160 80 {'Wallet': 37661.12017587529, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7289377289377289, 'No of Trade': 273}
160 90 {'Wallet': 32861.73821322642, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7323420074349443, 'No of Trade': 269}
160 100 {'Wallet': 39689.532864979294, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752851711026616, 'No of Trade': 263}
160 110 {'Wallet': 37343.17651974525, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752895752895753, 'No of Trade': 259}
160 120 {'Wallet': 34312.19714202723, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7529411764705882, 'No of Trade': 255}
170 10 {'Wallet': 13418.641738796012, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5498721227621484, 'No of Trade': 391}
170 20 {'Wallet': 28544.558337867416, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6330275229357798, 'No of Trade': 327}
170 30 {'Wallet': 34243.28391818847, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.670926517571885, 'No of Trade': 313}
170 40 {'Wallet': 40350.48748490615, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6897689768976898, 'No of Trade': 303}
170 50 {'Wallet': 34169.95306424348, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6949152542372882, 'No of Trade': 295}
170 60 {'Wallet': 34919.16292308454, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7142857142857143, 'No of Trade': 287}
170 70 {'Wallet': 34060.418774698934, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7208480565371025, 'No of Trade': 283}
170 80 {'Wallet': 37295.82443577217, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7289377289377289, 'No of Trade': 273}
170 90 {'Wallet': 32542.9942957429, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7323420074349443, 'No of Trade': 269}
170 100 {'Wallet': 39304.56244416981, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752851711026616, 'No of Trade': 263}
170 110 {'Wallet': 36980.96468852831, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752895752895753, 'No of Trade': 259}
170 120 {'Wallet': 33979.38443250005, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7529411764705882, 'No of Trade': 255}
180 10 {'Wallet': 12225.703138987848, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5419847328244275, 'No of Trade': 393}
180 20 {'Wallet': 27538.011145458357, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6299694189602446, 'No of Trade': 327}
180 30 {'Wallet': 33035.78647230953, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6677316293929713, 'No of Trade': 313}
180 40 {'Wallet': 38927.63590634855, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6864686468646864, 'No of Trade': 303}
180 50 {'Wallet': 32965.04143399649, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6915254237288135, 'No of Trade': 295}
180 60 {'Wallet': 33687.83241919397, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.710801393728223, 'No of Trade': 287}
180 70 {'Wallet': 32859.369577014695, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7208480565371025, 'No of Trade': 283}
180 80 {'Wallet': 36945.00597491245, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7289377289377289, 'No of Trade': 273}
180 90 {'Wallet': 32236.88273115585, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7323420074349443, 'No of Trade': 269}
180 100 {'Wallet': 38934.84904300409, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752851711026616, 'No of Trade': 263}
180 110 {'Wallet': 36633.10791610391, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.752895752895753, 'No of Trade': 259}
180 120 {'Wallet': 33659.76164555509, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7529411764705882, 'No of Trade': 255}
190 10 {'Wallet': 14326.457984449695, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.5473145780051151, 'No of Trade': 391}
190 20 {'Wallet': 29299.60536053393, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6338461538461538, 'No of Trade': 325}
190 30 {'Wallet': 35149.070907873815, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6720257234726688, 'No of Trade': 311}
190 40 {'Wallet': 41417.81930619443, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6910299003322259, 'No of Trade': 301}
190 50 {'Wallet': 35073.80034120725, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6962457337883959, 'No of Trade': 293}
190 60 {'Wallet': 35842.827941369585, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7157894736842105, 'No of Trade': 285}
190 70 {'Wallet': 34961.36870295538, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7259786476868327, 'No of Trade': 281}
190 80 {'Wallet': 39308.361427765114, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7343173431734318, 'No of Trade': 271}
190 90 {'Wallet': 34299.06165291275, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7378277153558053, 'No of Trade': 267}
190 100 {'Wallet': 41422.19267436273, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7586206896551724, 'No of Trade': 261}
190 110 {'Wallet': 38987.63320791423, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7587548638132295, 'No of Trade': 257}
190 120 {'Wallet': 35826.01666276498, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7628458498023716, 'No of Trade': 253}
200 20 {'Wallet': 28455.411230132275, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6338461538461538, 'No of Trade': 325}
200 30 {'Wallet': 34136.33920093184, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6720257234726688, 'No of Trade': 311}
200 40 {'Wallet': 40225.51701877971, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6910299003322259, 'No of Trade': 301}
200 50 {'Wallet': 34049.597660797874, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6962457337883959, 'No of Trade': 293}
200 60 {'Wallet': 35780.37569711259, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7157894736842105, 'No of Trade': 285}
200 70 {'Wallet': 34900.452305918625, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7259786476868327, 'No of Trade': 281}
200 80 {'Wallet': 39239.87086688515, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7343173431734318, 'No of Trade': 271}
200 90 {'Wallet': 34239.29925415238, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7378277153558053, 'No of Trade': 267}
200 100 {'Wallet': 41350.01899156097, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7586206896551724, 'No of Trade': 261}
200 110 {'Wallet': 38919.7014812076, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7587548638132295, 'No of Trade': 257}
200 120 {'Wallet': 35763.59371033957, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7628458498023716, 'No of Trade': 253}
200 130 {'Wallet': 41466.62037584262, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7714285714285715, 'No of Trade': 245}
210 20 {'Wallet': 29942.50331622537, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6386292834890965, 'No of Trade': 321}
210 30 {'Wallet': 35920.3190374328, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6775244299674267, 'No of Trade': 307}
210 40 {'Wallet': 42856.98671385047, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.696969696969697, 'No of Trade': 297}
210 50 {'Wallet': 36277.05155112105, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7024221453287197, 'No of Trade': 289}
210 60 {'Wallet': 38121.05348830781, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7224199288256228, 'No of Trade': 281}
210 70 {'Wallet': 37178.08890519463, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7328519855595668, 'No of Trade': 277}
210 80 {'Wallet': 41800.70203474151, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7415730337078652, 'No of Trade': 267}
210 90 {'Wallet': 36473.78837856974, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7452471482889734, 'No of Trade': 263}
210 100 {'Wallet': 44048.56042622221, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7665369649805448, 'No of Trade': 257}
210 110 {'Wallet': 41459.63809146939, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.766798418972332, 'No of Trade': 253}
210 120 {'Wallet': 38097.55973583129, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7710843373493976, 'No of Trade': 249}
210 130 {'Wallet': 44172.77132736749, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7800829875518672, 'No of Trade': 241}
220 20 {'Wallet': 32708.588432705103, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6394984326018809, 'No of Trade': 319}
220 30 {'Wallet': 39238.63410345478, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6786885245901639, 'No of Trade': 305}
220 40 {'Wallet': 46816.11036608422, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6983050847457627, 'No of Trade': 295}
220 50 {'Wallet': 39628.32152696674, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7038327526132404, 'No of Trade': 287}
220 60 {'Wallet': 41642.67215742523, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7240143369175627, 'No of Trade': 279}
220 70 {'Wallet': 40276.403721822346, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7345454545454545, 'No of Trade': 275}
220 80 {'Wallet': 45284.252111506816, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7433962264150943, 'No of Trade': 265}
220 90 {'Wallet': 39513.408818448625, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7471264367816092, 'No of Trade': 261}
220 100 {'Wallet': 47719.44054509281, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7686274509803922, 'No of Trade': 255}
220 110 {'Wallet': 44914.76488183195, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7689243027888446, 'No of Trade': 251}
220 120 {'Wallet': 41272.50060242301, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7732793522267206, 'No of Trade': 247}
220 130 {'Wallet': 47854.00282487897, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7824267782426778, 'No of Trade': 239}
230 20 {'Wallet': 35933.86702310972, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6435331230283912, 'No of Trade': 317}
230 30 {'Wallet': 43107.81747561321, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6831683168316832, 'No of Trade': 303}
230 40 {'Wallet': 51432.48195791871, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7030716723549488, 'No of Trade': 293}
230 50 {'Wallet': 43535.93060210481, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7087719298245614, 'No of Trade': 285}
230 60 {'Wallet': 45748.90924658953, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7328519855595668, 'No of Trade': 277}
230 70 {'Wallet': 40415.157356172735, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7345454545454545, 'No of Trade': 275}
230 80 {'Wallet': 45440.25796055691, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7433962264150943, 'No of Trade': 265}
230 90 {'Wallet': 39649.53390838948, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7471264367816092, 'No of Trade': 261}
230 100 {'Wallet': 48380.73679248559, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7686274509803922, 'No of Trade': 255}
230 110 {'Wallet': 45537.193919758756, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7689243027888446, 'No of Trade': 251}
230 120 {'Wallet': 41844.45512362297, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7732793522267206, 'No of Trade': 247}
230 130 {'Wallet': 48517.163837022745, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7824267782426778, 'No of Trade': 239}
240 40 {'Wallet': 51803.99328550461, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7030716723549488, 'No of Trade': 293}
240 100 {'Wallet': 48730.204503767, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7686274509803922, 'No of Trade': 255}
250 40 {'Wallet': 50047.87371268597, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6996587030716723, 'No of Trade': 293}
250 100 {'Wallet': 47078.28424648327, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7647058823529411, 'No of Trade': 255}
260 40 {'Wallet': 52039.59969010565, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7010309278350515, 'No of Trade': 291}
260 100 {'Wallet': 48951.831207626274, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.766798418972332, 'No of Trade': 253}
270 40 {'Wallet': 50867.22434060158, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7010309278350515, 'No of Trade': 291}
270 100 {'Wallet': 47849.01872323622, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.766798418972332, 'No of Trade': 253}
280 40 {'Wallet': 51471.402331400444, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7010309278350515, 'No of Trade': 291}
280 100 {'Wallet': 48417.34782647811, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.766798418972332, 'No of Trade': 253}
290 40 {'Wallet': 51510.35525352979, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7010309278350515, 'No of Trade': 291}
290 100 {'Wallet': 48417.34782647811, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.766798418972332, 'No of Trade': 253}
300 40 {'Wallet': 52377.739118738005, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.7010309278350515, 'No of Trade': 291}
300 100 {'Wallet': 49232.64847999053, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.766798418972332, 'No of Trade': 253}
310 40 {'Wallet': 50984.930268248085, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.697594501718213, 'No of Trade': 291}
320 40 {'Wallet': 50854.43210430824, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.697594501718213, 'No of Trade': 291}
330 40 {'Wallet': 51082.92755063437, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.697594501718213, 'No of Trade': 291}
340 40 {'Wallet': 51082.92755063437, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.697594501718213, 'No of Trade': 291}
350 40 {'Wallet': 47464.331487370444, 'Hodling Wallet': 17007.172532050543, 'Win Probability': 0.6928327645051194, 'No of Trade': 293}
"""                    
