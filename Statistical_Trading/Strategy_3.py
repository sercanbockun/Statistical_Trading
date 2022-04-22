# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 00:31:54 2022

@author: serca
"""
#from WVMA_Strategy import ema, fisher_transform, rsi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import json
import warnings

warnings.filterwarnings("ignore")
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
    return df

def rsi(df, start_time, end_time, period, show_on_graph):
    difference = df.loc[start_time : end_time, "Close"].diff(1)
    difference = difference.dropna()
    up_days = difference.copy()
    down_days = difference.copy()
    up_days[up_days<0] = 0
    down_days[down_days>0] = 0
    
    avg_up = up_days.rolling(window = period).mean()
    avg_down = abs(down_days.rolling(window = period).mean())
    rs = avg_up/avg_down
    rsi= 100- (100/(1+rs))
    
    if show_on_graph:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(rsi, linewidth = 3, alpha = 0.8)
        plt.grid()
    return rsi

def ema(values, period):
    return values.ewm(span=period).mean()

def trade(orig_df, start_time, end_time, ema_period, long, short, show_on_graph, fee): 

    
    #long, short, show_on_graph : BOOLEAN
    df = orig_df.copy(deep = True)
    index1 = "EMA" + str(ema_period)
    df[index1] = ema(df["Close"], ema_period)
    df = fisher_transform(df, 100, False)

    rsi_ = rsi(df, start_time, end_time, 14, False)
 
    
    cross_df = pd.DataFrame()
    cross_df["Close"] = df.loc[start_time : end_time, "Close"]

    cross_df[index1] = df.loc[start_time : end_time, index1]
    cross_df["Volume"] = df.loc[start_time : end_time, "Volume"]
    cross_df["Price Fisher"] = df.loc[start_time : end_time, "Fisher"]




    
    
    wallet = 10000
    buy_points = []
    sell_points = []
    buy_volumes = []
    sell_volumes = []
    cond = 0
    first_changer = 0
    time_in_trade = 1
    times_in_trades = []


    for i in range(0,len(cross_df)):

#double_ma_trade(df, x, y, start_time, end_time, mov_avg_type, long, short, show_on_graph, fee, ema_fisher_period): 

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
            
    trades_df = pd.DataFrame()
    trades_df["Buy Volumes"] = buy_volumes
    trades_df["Sell Volumes"] = sell_volumes
    trades_df["Buy Points"] = buy_points
    trades_df["Sell Points"] = sell_points
    trades_df["Win Situation"] = np.nan
    trades_df["Return on Trade"] = np.nan
    trades_df["Max Drawdown"] = np.nan
    trades_df["Max Drawup"] = np.nan
    trades_df["Volumes in Trade"] = np.nan
    trades_df["Price Fisher"] =  cross_df["Price Fisher"].values.tolist()
    cross_df["RSI"] = rsi(df, start_time, end_time, 14, False)
    trades_df["RSI_"] = cross_df["RSI"].values.tolist()
    temporary_df = pd.DataFrame()
    temporary_df["Close"] = cross_df[index1]
    trades_df[str(index1 + "Fisher(50)")] = (fisher_transform(temporary_df, 50, False))["Fisher"].values.tolist()

    
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
            trades_df["Return on Trade"][idx_1] = fee*(every[1]-every[0])*100/every[0]
            
        elif every[1] < every[0]:
            trades_df["Win Situation"][idx_1] = -1      
            trades_df["Return on Trade"][idx_1] = (1 + (1- fee))*(every[1]-every[0])*100/every[0]
        

        try:
            max_value = max(cross_df.loc[start : end ,"Close"])
            min_value = min(cross_df.loc[start : end ,"Close"])
            bar_volumes = (cross_df.loc[start : end ,"Volume"])

            max_return = 100 * (max_value - every[0] )/every[0]
            min_return = 100 * (min_value - every[0] )/every[0]

            max_returns_for_long_trades.append(max_return)
            min_returns_for_long_trades.append(min_return)
            
        except:
            start = (cross_df[cross_df["Close"] == every[0]].index.values)[0]
            end = (cross_df[cross_df["Close"] == every[1]].index.values)[1]
     
            max_value = max(cross_df.loc[start : end ,"Close"])
            min_value = min(cross_df.loc[start : end ,"Close"])
            bar_volumes = (cross_df.loc[start : end ,"Volume"])  

            max_return = 100 * (every[0] - min_value)/every[0]
            min_return = 100 * (every[0] - max_value)/every[0]
            
            max_returns_for_long_trades.append(max_return)
            min_returns_for_long_trades.append(min_return)

        trades_df["Max Drawdown"][idx_1] =  min_return
        trades_df["Max Drawup"][idx_1] =  max_return    
        #trades_df["Volumes in Trade"][idx_1] =  [dict(bar_volumes)]
    
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
            bar_volumes = (cross_df.loc[start : end ,"Volume"])

            max_return = 100 * (every[0] - min_value)/every[0]
            min_return = 100 * (every[0] - max_value)/every[0]

            max_returns_for_short_trades.append(max_return)
            min_returns_for_short_trades.append(min_return)
            
        except:
            start = (cross_df[cross_df["Close"] == every[0]].index.values)[0]
            end = (cross_df[cross_df["Close"] == every[1]].index.values)[1]
            bar_volumes = (cross_df.loc[start : end ,"Volume"])
            
            max_value = max(cross_df.loc[start : end ,"Close"])
            min_value = min(cross_df.loc[start : end ,"Close"])

            max_return = 100 * (every[0] - min_value)/every[0]
            min_return = 100 * (every[0] - max_value)/every[0]
            
            max_returns_for_short_trades.append(max_return)
            min_returns_for_short_trades.append(min_return)
            
        trades_df["Max Drawdown"][idx_1] =  min_return
        trades_df["Max Drawup"][idx_1] =  max_return 
        #trades_df["Volumes in Trade"][idx_1] =  [dict(bar_volumes)]
        
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



#print(trade(df_1h, end_time_1h - 300*day, end_time_1h - 100*day ,270, True, True, False, 0.99925)[1])
#result = (trade(df_1h, end_time_1h-100*day, end_time_1h, 270, True, True, True, 0.99925)[0])
#print(result["Wallet"],result["Hodling Wallet"],result["Win Probability"], result["No of Trade"])

# for i in range(100,400,10):
#     result = (trade(df_1h, end_time_1h-200*day, end_time_1h, i, True, True, False, 0.99925)[0])
#     print(i, result["Win Probability"], result["No of Trade"])
