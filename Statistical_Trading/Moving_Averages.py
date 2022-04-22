#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import json

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
def ema(values, period):
    return values.ewm(span=period).mean()

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
fisher_transform(df_1h, 70, True )





def double_ma_trade(df, x, y, start_time, end_time, mov_avg_type, long, short, show_on_graph, fee, ema_fisher_period): 
    #   (df, int,int, datetime, datetime, 
    # MOV_AVG_TYPE : int (1- (SMA-SMA), 2- (EMA-EMA), 3- (EMA-SMA))
    #long, short, show_on_graph : BOOLEAN
    if mov_avg_type == 1:
        index1 = "SMA" + str(x)
        index2 = "SMA" + str(y) 
        df[index1] = df["Close"].rolling(window = x).mean()
        df[index2] = df["Close"].rolling(window = y).mean()  
        
    if mov_avg_type == 2:
        index1 = "EMA" + str(x)
        index2 = "EMA" + str(y)
        df[index1] = ema(df["Close"], x)
        df[index2] = ema(df["Close"], y)
        
    if mov_avg_type == 3:
        index1 = "EMA" + str(x)
        index2 = "SMA" + str(y)    
        df[index1] = ema(df["Close"], x)
        df[index2] = df["Close"].rolling(window = y).mean()    
        
    if index1 == index2 :
        return {"Hodling Wallet": 0,"Wallet":0, "Total Return": 0, "Buy Points": 0, "Sell Points" : 0, "Long Trades": 0, "Short Trades":0  , "Win Probability" : 0 ,str(index1):None,  str(index2):None , "No of Trade" : 0}
     
    rsi_ = rsi(df, start_time, end_time, 14, False)
    
    cross_df = pd.DataFrame()
    cross_df[index1] = df.loc[start_time : end_time, index1]
    cross_df[index2] = df.loc[start_time : end_time, index2]
    cross_df["Close"] = df.loc[start_time : end_time, "Close"]
    cross_df["Fisher"] = df.loc[start_time : end_time, "Fisher"]
    cross_df["Fisher MA"] = df.loc[start_time : end_time, "Fisher MA"]
    cross_df["RSI"] = rsi_
    
    temporary_df = pd.DataFrame()
    temporary_df["Close"] = cross_df[index1]
    cross_df[str(index1 + "Fisher")] = (fisher_transform(temporary_df, ema_fisher_period, False))["Fisher"]

    wallet = 10000
    wallet2 = 10000
    buy_points = []
    sell_points = []
    cond = 0
    first_changer = 0
    

    for i in range(0,len(cross_df)):

        if cross_df.iloc[i,][index1]-cross_df.iloc[i,][index2] > 0 and cond != 1 and not cross_df.iloc[i,][str(index1 + "Fisher")] < -2: 
   
            if first_changer == 0:
                first_changer = 1
            cond = 1
            buy_points.append(cross_df.iloc[i,]["Close"])
            sell_points.append(np.nan)
        elif (cross_df.iloc[i,][index1] - cross_df.iloc[i,][index2]) < 0 and cond != -1     and not (cross_df.iloc[i,][str(index1 + "Fisher")] > 2):

            if first_changer == 0:
                first_changer = -1
            cond = -1
            sell_points.append(cross_df.iloc[i,]["Close"])
            buy_points.append(np.nan)
        else:
            buy_points.append(np.nan)
            sell_points.append(np.nan)

    sell_points = pd.Series(sell_points).fillna(0)
    buy_points= pd.Series(buy_points).fillna(0)
    
    geniune_sell_points = []
    geniune_buy_points = []
    
    for every in sell_points:
        if every != 0:
            geniune_sell_points.append(every)
    for every in buy_points:
        if every != 0:
            geniune_buy_points.append(every)
    
    
    if first_changer == 1:

        if len(geniune_buy_points) > len(geniune_sell_points):
            geniune_sell_points.append(cross_df.iloc[-1,]["Close"])
            
        for k in range(0,len(geniune_sell_points)-1):
            if long == True:
                wallet += wallet*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_buy_points[k]
                wallet *= fee
            if short == True:
                wallet += wallet*(geniune_sell_points[k]-geniune_buy_points[k+1])/geniune_sell_points[k]
                wallet *= fee
            
            if long == False:
                wallet2 += wallet2*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_buy_points[k]
                wallet2 *= fee
            if short == False:
                wallet2 += wallet2*(geniune_sell_points[k]-geniune_buy_points[k+1])/geniune_sell_points[k]
                wallet2 *= fee
            
                
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
                    
                if short == False:
                    wallet2 += wallet2*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_sell_points[k]
                    wallet2 *= fee
                if long == False:
                    wallet2 += wallet2*(geniune_sell_points[k+1]-geniune_buy_points[k])/geniune_buy_points[k]
                    wallet2 *= fee

        elif len(geniune_sell_points) > len(geniune_buy_points):
            geniune_buy_points.append(cross_df.iloc[-1,]["Close"])
            for k in range(0,len(geniune_sell_points)-1):
                if short == True:
                    wallet += wallet*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_sell_points[k]
                    wallet *= fee
                if long == True:
                    wallet +=  wallet*((geniune_sell_points[k+1]-geniune_buy_points[k])/geniune_buy_points[k])
                    wallet *= fee
                    
                if short == False:
                    wallet2 += wallet2*(geniune_sell_points[k]-geniune_buy_points[k])/geniune_sell_points[k]
                    wallet2 *= fee
                if long == False:
                    wallet2 += wallet2*(geniune_sell_points[k+1]-geniune_buy_points[k])/geniune_buy_points[k]
                    wallet2 *= fee
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
        
    if show_on_graph == True:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,15))
        plt.plot(cross_df["Close"], linewidth = 3, alpha = 0.2)
        plt.plot(cross_df[index1], linewidth=1, alpha = 0.5)
        plt.plot(cross_df[index2], linewidth=2, alpha = 0.5)
        plt.grid()

        plt.scatter(cross_df.index, buy_points, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points, marker = '1', color = 'red')

        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(cross_df[index1 + "Fisher"], linewidth = 3, alpha = 0.8)
        plt.scatter(cross_df.index, buy_points/9999999999, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points/9999999999, marker = '1', color = 'red')
        plt.grid()        
        
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(cross_df["Fisher"], linewidth = 3, alpha = 0.8)
        plt.plot(cross_df["Fisher MA"], linewidth = 3, alpha = 0.8)
        plt.scatter(cross_df.index, buy_points/9999999999, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points/9999999999, marker = '1', color = 'red')
        plt.grid()
        
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(cross_df["RSI"], linewidth = 3, alpha = 0.8)
        plt.scatter(cross_df.index, buy_points/9999999999, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points/9999999999, marker = '1', color = 'red')
        plt.grid()        
        

        #plt.scatter(cross_df.index, buy_points, marker = '2', color = 'green')
        #plt.scatter(cross_df.index, sell_points, marker = '1', color = 'red')
        
    if long == True and short == False:
        return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return,# "Long Trades": trade_open_close_long, 
                "Win Probability" : win_prob , "No of Trade" : num_of_trade}
    
    if long == False and short == True:
        return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return, # "Short Trades": trade_open_close_short  ,
                "Win Probability" : win_prob , "No of Trade" : num_of_trade}
    
    else:
        return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return, #"Long Trades": trade_open_close_long, "Short Trades": trade_open_close_short  , 
                "Win Probability" : win_prob , "No of Trade" : num_of_trade}



print(double_ma_trade(df_1h,20,20, end_time_1h - 800*day, end_time_1h - 600* day, 3, True, True, True, 0.99925, 170 ))

"""def conf_interval( trade_time_interval, x, y, mov_avg_type, long, short, confidentiality):    
  
    time_ranges = []
    end_time = start_time +datetime.timedelta(days =trade_time_interval) 
    time_ranges.append([start_time, end_time])
    interval = trade_time_interval
    
    while interval < 720:
        end_time = time_ranges[-1][-1] +datetime.timedelta(days =trade_time_interval) 
        time_ranges.append([time_ranges[-1][-1], end_time ])
        interval += trade_time_interval
    time_ranges[-1][1] = datetime.datetime.now()- timedelta(days =1)
    
    wallets = []
    hodl_wallets = []
    win_probs = []
    
    highest_trade = {"Trade time": 0, "Highest Wallet": 0}
    lowest_trade = {"Trade time": 0, "Lowest Wallet": 10000}
 
    for every in time_ranges:

        start = every[0]
        end = every[1]
        
        trade1 = trade(data, x, y, start, end, mov_avg_type, long, short, False, 0.99925 )

        wallet = (trade1["Wallet"])
        wallets.append(wallet)
                            
        win_probs.append(trade1["Win Probability"]) 
    
        if wallet > highest_trade["Highest Wallet"]:
                            
            highest_trade["Highest Wallet"] = wallet
            highest_trade["Trade time"] = [str(i.date()) for i in every]
        
        elif wallet < lowest_trade["Lowest Wallet"]:
                            
            lowest_trade["Lowest Wallet"] = wallet
            lowest_trade["Trade time"] = [str(i.date()) for i in every]
        
        hodl_wallets.append(trade1["Hodling Wallet"])
        
          
                            

    mean_wallet = sum(wallets) / len(wallets)
    mean_win_prob = sum(win_probs) / len(wallets)
    
    std_of_mean_wallet = (sum([((x - mean_wallet) ** 2) for x in wallets]) / len(wallets))**(0.5)
    std_of_mean_win_prob = (sum([((x - mean_win_prob) ** 2) for x in win_probs]) / len(wallets))**(0.5)
    
    half_length1 = ((t.ppf( confidentiality + ((1-confidentiality)/2), len(wallets) - 1 ))*std_of_mean_wallet)/((len(wallets))**(0.5))
    CI_of_Mean_Wallet = (mean_wallet - half_length1 , mean_wallet + half_length1)
    
    half_length2 = ((t.ppf( confidentiality + ((1-confidentiality)/2), len(wallets) - 1 ))*std_of_mean_win_prob)/((len(wallets))**(0.5))
    CI_of_Mean_Win_Prob = (mean_win_prob - half_length2, mean_win_prob + half_length2 )
    
    mean_hodl_wallet = sum(hodl_wallets) / len(hodl_wallets)
    
    return {"Mean Wallet": (mean_wallet),  "CI of Mean of Return": CI_of_Mean_Wallet,
            "Mean Win Probability": mean_win_prob, "CI of Win Probability": CI_of_Mean_Win_Prob,
             "Highest Trade" : highest_trade, "Lowest Trade": lowest_trade,
            "Mean Hodl Wallet":  mean_hodl_wallet }
                            




def find_best_pair(trade_time_interval, mov_avg_type, long, short):
    if mov_avg_type == 1:
        index1 = "SMA" 
        index2 = "SMA" 
    if mov_avg_type == 2:
        index1 = "EMA" 
        index2 = "EMA" 
    if mov_avg_type == 3:
        index1 = "EMA" 
        index2 = "SMA"   
    
    pair_type = str(index1 + " - " + index2)
    highest_mean_wallet = {"Wallet": 0, pair_type : []}
    highest_mean_win_prob = {"Win Probability": 0, pair_type : []}
    for i in range (16,24,1):
        for j in range (i,25,1):
            interv = conf_interval(trade_time_interval, i, j, mov_avg_type, long, short, 0.7)["Mean Wallet"]
            if interv > highest_mean_wallet["Wallet"]:
                highest_mean_wallet["Wallet"] =  interv
                if len(highest_mean_wallet[pair_type])<2:

                    highest_mean_wallet[pair_type].append(i)
                    highest_mean_wallet[pair_type].append(j)

                elif len(highest_mean_wallet[pair_type])==2:
                    highest_mean_wallet[pair_type][0] = i
                    highest_mean_wallet[pair_type][1] = j
            win_proba = conf_interval(30, i, j, mov_avg_type, long, short, 0.7)["Mean Win Probability"]
            if  win_proba > highest_mean_win_prob["Win Probability"]:
                highest_mean_win_prob["Win Probability"] =  win_proba
                if len(highest_mean_win_prob[pair_type])<2:

                    highest_mean_win_prob[pair_type].append(i)
                    highest_mean_win_prob[pair_type].append(j)

                elif len(highest_mean_win_prob[pair_type])==2:
                    highest_mean_win_prob[pair_type][0] = i
                    highest_mean_win_prob[pair_type][1] = j
    return [highest_mean_wallet, highest_mean_win_prob]


"""



