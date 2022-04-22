#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import json


f_1h = open('Binance_1h_BTCUSDT_Candles.json',)
f_4h = open('Binance_4h_BTCUSDT_Candles.json',)
f_1d = open('Bitfinex_1D_tBTCUSD_Candles.json',)
data_bin_1h = json.load(f_1h)
data_bin_4h = json.load(f_4h)
data_bin_1d = json.load(f_1d)

df_1h = pd.DataFrame(data_bin_1h,columns=["Open time","Open","High","Low", "Close","Volume", 
                                                  "Close time","Quote asset volume","Number of trades", 
                                                  "Taker buy base asset volume", "Taker buy quote asset volume","Ignore" ]).astype(float)

df_4h = pd.DataFrame(data_bin_4h,columns=["Open time","Open","High","Low", "Close","Volume", 
                                                  "Close time","Quote asset volume","Number of trades", 
                                                  "Taker buy base asset volume", "Taker buy quote asset volume","Ignore" ]).astype(float)

df_1d = pd.DataFrame(data_bin_1d,columns=["Close time","Open","Close","High","Low", "Volume" ]).astype(float)

df_1h = df_1h.set_index("Close time")
df_1h = df_1h.drop(columns = ["Ignore","Taker buy base asset volume", "Taker buy quote asset volume" ])
df_4h = df_4h.set_index("Close time")
df_4h = df_4h.drop(columns = ["Ignore","Taker buy base asset volume", "Taker buy quote asset volume" ])
df_1d = df_1d.set_index("Close time")
start_time_1h = df_1h.index[0]
end_time_1h = df_1h.index[-1]
day = 86400000




def ema(values, period):
    return values.ewm(span=period).mean()

def VWMA(df, period,show_on_graph):
    value_list = []
    prices = df["Close"]
    volumes = df["Volume"]
    multiplied_values = prices.multiply(volumes)            
    for i in range(0, len(df)):
        if i< period-1:
            value_list.append(np.nan)
        else:
            #prices = df.iloc[i-period+1 : i+1, :  ]["Close"]
            #volumes = df.iloc[i-period+1 : i+1, :  ]["Volume"]
            #value_list.append((prices.multiply(volumes).sum())/(volumes.sum()))
            value_list.append(multiplied_values.iloc[i-period+1 : i+1].sum() / df.iloc[i-period+1 : i+1, :  ]["Volume"].sum())


    df["VWMA"] = value_list
    if show_on_graph:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(df["VWMA"], linewidth = 3, alpha = 0.8)
        plt.grid()
    return df
     

def fisher_transform(df, period, show_on_graph):
    fisher_liste = []

        
    for i in range(0, len(df)):
        if i< period-1:
            fisher_liste.append(np.nan)

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


    df["Fisher"] = fisher_liste
    df["Fisher Previous"] = df["Fisher"].shift(periods = 1, fill_value = 0)
    #df["Fisher MA"] = df["Fisher"].rolling(window = 7).mean() 
    
    
    if show_on_graph:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(df["Fisher"], linewidth = 3, alpha = 0.8)
        plt.plot(df["Fisher Previous"], linewidth = 3, alpha = 0.8)
        #plt.plot(df["Fisher MA"], linewidth = 3, alpha = 0.8)
        plt.grid()
    return df



def rsi(df, start_time, end_time, period, show_on_graph):
    difference = df.loc[start_time:end_time, "Close"].diff(1)
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

VWMA(df_1d, 20,False)
fisher_transform(df_1d, 14, False )

VWMA(df_4h, 20,False)
fisher_transform(df_4h, 14,False )


# In[19]:


def trade(orig_df, start_time, end_time, long, short, show_on_graph, fee, vwma_period, vwma_fisher_period, percentage_differ): 
    #   (df, datetime, datetime, 

    #long, short, show_on_graph : BOOLEAN
    df = orig_df.copy(deep = True)
    VWMA(df, vwma_period,False)
    #rsi_ = rsi(df, start_time, end_time, 14, False)
    cross_df = pd.DataFrame()
    cross_df["Close"] = df.loc[start_time : end_time, "Close"]
    #cross_df["Fisher"] = df.loc[start_time : end_time, "Fisher"]
    #cross_df["RSI"] = rsi_
    cross_df["VWMA"] = df.loc[start_time:end_time, "VWMA"]
    
    temporary_df = pd.DataFrame()
    temporary_df["Close"] = cross_df["VWMA"]
    cross_df[str("VWMA" + " Fisher")] = (fisher_transform(temporary_df, vwma_fisher_period, False))["Fisher"]
    
    r = df.loc[start_time : end_time, "Close"]
    b = df.loc[start_time : end_time, "Close"]
    wallet = 10000
    wallet2 = r
    hodlwallet2 = b.multiply(10000/r.iloc[0])
    
    
    buy_points = []
    sell_points = []
    cond = 0
    first_changer = 0


    for i in range(0,len(cross_df)):
        
        if i == 0:
            wallet2.iloc[i] = 10000
#        if cond == 1:
#            wallet2.iloc[i] = wallet2.iloc[i-1] + wallet2.iloc[i-1] * (cross_df.iloc[i,]["Close"] - cross_df.iloc[i-1,]["Close"])/cross_df.iloc[i-1,]["Close"]
#        if cond == -1:
#            wallet2.iloc[i] = wallet2.iloc[i-1] + wallet2.iloc[i-1] * (cross_df.iloc[i-1,]["Close"] - cross_df.iloc[i,]["Close"])/cross_df.iloc[i-1,]["Close"]
        
        if cross_df.iloc[i,]["Close"]> ((1+percentage_differ)* cross_df.iloc[i,]["VWMA"])  and cond != 1             and not cross_df.iloc[i,]["VWMA" + " Fisher"] < -3 :  
            
            if first_changer == 0:
                first_changer = 1
            cond = 1
            buy_points.append(cross_df.iloc[i,]["Close"])
            sell_points.append(np.nan)
            if i != 0:
                wallet2.iloc[i] = wallet2.iloc[i-1] * fee
        elif (cross_df.iloc[i,]["Close"] < (1 - percentage_differ) * cross_df.iloc[i,]["VWMA"]) and cond != -1             and not (cross_df.iloc[i,]["VWMA" + " Fisher"] > 3):
            
            if first_changer == 0:
                first_changer = -1
            cond = -1
            sell_points.append(cross_df.iloc[i,]["Close"])
            buy_points.append(np.nan)
            if i != 0:
                wallet2.iloc[i] = wallet2.iloc[i-1] * fee
        else:
            buy_points.append(np.nan)
            sell_points.append(np.nan)

#            if cond == 1:
#                wallet2.iloc[i] = wallet2.iloc[i-1] + wallet2.iloc[i-1] * (cross_df.iloc[i,]["Close"] - cross_df.iloc[i-1,]["Close"])/cross_df.iloc[i-1,]["Close"]
#            elif cond == -1:
#                wallet2.iloc[i] = wallet2.iloc[i-1] + wallet2.iloc[i-1] * (cross_df.iloc[i-1,]["Close"] - cross_df.iloc[i,]["Close"])/cross_df.iloc[i-1,]["Close"]
      
            
    
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
    
    
    zipped_returns_long = zip(long_trade_performances , max_returns_for_long_trades, min_returns_for_long_trades)
    
    
    max_returns_for_short_trades =[]
    min_returns_for_short_trades = []
    for every in trade_open_close_short:
        start = (cross_df[cross_df["Close"] == every[0]].index.values)[0]
        end = (cross_df[cross_df["Close"] == every[1]].index.values)[0]
        
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
    
        
    if show_on_graph == True:
        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,15))
        plt.plot(cross_df["Close"], linewidth = 3, alpha = 0.2)
        plt.plot(cross_df["VWMA"], linewidth = 3, alpha = 0.8)
        plt.grid()
        plt.scatter(cross_df.index, buy_points, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points, marker = '1', color = 'red')

        plt.style.use('seaborn-bright')
        plt.figure(figsize= (120,5))
        plt.plot(cross_df["VWMA" + " Fisher"], linewidth = 3, alpha = 0.8)
        plt.scatter(cross_df.index, buy_points/9999999999, marker = '2', color = 'green')
        plt.scatter(cross_df.index, sell_points/9999999999, marker = '1', color = 'red')
        plt.grid()        
        
#        plt.style.use('seaborn-bright')
#        plt.figure(figsize= (120,5))
#        plt.plot(cross_df["Fisher"], linewidth = 3, alpha = 0.8)
#        plt.scatter(cross_df.index, buy_points/9999999999, marker = '2', color = 'green')
#        plt.scatter(cross_df.index, sell_points/9999999999, marker = '1', color = 'red')
#        plt.grid()
        
          
#        plt.style.use('seaborn-bright')
#        plt.figure(figsize= (120,5))
#        plt.plot(wallet2, linewidth = 3, alpha = 0.8)
#        plt.plot(hodlwallet2, linewidth = 3, alpha = 0.8)
#        plt.scatter(cross_df.index, buy_points/9999999999, marker = '2', color = 'green')
#        plt.scatter(cross_df.index, sell_points/9999999999, marker = '1', color = 'red')
#        plt.grid()

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
                "Win Probability" : win_prob , "No of Trade" : num_of_trade}
    
    if long == False and short == True:
        return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return,  "Short Trades": trade_open_close_short  ,
                "Win Probability" : win_prob , "No of Trade" : num_of_trade}

    else:
        return {"Wallet":wallet, "Hodling Wallet": hodling_wallet, "Total Return": total_return, 
                "Long Trades": trade_open_close_long, "Short Trades": trade_open_close_short  , 
                "Win Probability" : win_prob , "No of Trade" : num_of_trade, "Trades' Return Mean" : total_trades_return_mean,
                "%90 Percent T Distribution Confidence Interval for Trades' Return Mean" : conf_interval,
               "Long Trades' Return Mean" : long_return_mean, "Short Trades' Return Mean" : short_return_mean,
               "Winning Trade's Return Mean" : mean_of_winning_trades, "Losing Trade's Return Mean" : mean_of_losing_trades,
               "Weighted Ratio of Winning Trades Return Mean to Losing Trades Return Mean" : abs(ratio_win_to_lose_return_mean)}


# In[3]:


# start_time = df_4h.index[0]
# end_time = df_4h.index[-1]
# start = time.time()
# trade(df_4h, end_time - 720*day, end_time, True, True, False, 0.99925, 20,30,0.025)
# end = time.time()
# print(end-start)


# # In[85]:


# end_time = df_1h.index[-1]
# VWMA(df_1h, 20,False)
# fisher_transform(df_1h, 14,False )


# # In[94]:


# trade(df_1h, end_time - 40*day, end_time, True, False, True, 0.99925)


# # In[184]:


# trade(df_1d, start_time, end_time, True, False, True, 0.99925)


# # In[185]:


# trade(df_1d, start_time, end_time, True, True, True, 0.99925)


# # In[187]:


# trade(df_1d, start_time, end_time, True, True, True, 0.99925)


# # In[190]:


# trade(df_1d, end_time- 200*day, end_time, True, True, True, 0.99925)


# # In[189]:


# trade(df_1d, start_time, end_time, True, True, True, 1)


# # In[ ]:





# # In[144]:


# end_time = df_4h.index[-1]

# brute_force_search_results = {}
# for vwma_period in range(5,100,5):
#     for vwma_fisher_period in range(5,100,5):

#         key_name =  str(vwma_period) + " / " + str(vwma_fisher_period) + " / " + str(percentage_differ)
#         results = trade(df_4h, end_time- 720*day, end_time - 500*day, True, True, False, 0.99925, vwma_period, vwma_fisher_period, 0)
#         wanted = ["Wallet", "Win Probability", "Trades' Return Mean",  "Weighted Ratio of Winning Trades Return Mean to Losing Trades Return Mean"]
#         values = [results.get(k) for k in wanted]
#         brute_force_search_results[key_name] = values
#         if  vwma_fisher_period == 95:
#             print(brute_force_search_results)
            
# print(brute_force_search_results)


# # In[4]:


# brute_force_search_results = {'5 / 5 / 1': [27187.919244106215, 0.3230088495575221, 0.5865311076400381, 1.833060280522151], '5 / 10 / 1': [27457.590752228567, 0.3282442748091603, 0.5192662553987631, 1.829955990476319], '5 / 15 / 1': [30187.063430680835, 0.337037037037037, 0.5415792977031385, 1.8905894903897], '5 / 20 / 1': [31666.821075075066, 0.3464285714285714, 0.5422867095080942, 1.8956091520894147], '5 / 25 / 1': [27590.941420684245, 0.336734693877551, 0.4729730508814969, 1.7646717002645462], '5 / 30 / 1': [25549.989435591087, 0.3355263157894737, 0.4340431493193248, 1.7054137418195594], '5 / 35 / 1': [25192.96981232591, 0.3419354838709677, 0.4225441551176507, 1.6968312424980103], '5 / 40 / 1': [24040.249160258, 0.3375796178343949, 0.4034677220735822, 1.6560193168251947], '5 / 45 / 1': [28515.695140426196, 0.34177215189873417, 0.455875937349557, 1.774594704799186], '5 / 50 / 1': [28047.011488144733, 0.33962264150943394, 0.4482850971783213, 1.760376155073466], '5 / 55 / 1': [29249.818668197535, 0.3385093167701863, 0.4573632311049458, 1.7723538056579746], '5 / 60 / 1': [29048.227357620483, 0.3395061728395062, 0.4528925447879246, 1.7672348444432493], '5 / 65 / 1': [29048.227357620483, 0.3395061728395062, 0.4528925447879246, 1.7672348444432493], '5 / 70 / 1': [29553.573215014174, 0.3395061728395062, 0.45821934021198607, 1.7797797356918137], '5 / 75 / 1': [29397.569251350014, 0.3395061728395062, 0.45648359750340767, 1.7768259166978397], '5 / 80 / 1': [30259.640399210537, 0.3425925925925926, 0.4657250669242516, 1.7870196297180665], '5 / 85 / 1': [30259.640399210537, 0.3425925925925926, 0.4657250669242516, 1.7870196297180665], '5 / 90 / 1': [30050.81380157793, 0.34049079754601225, 0.46123608254300663, 1.7809922034418375], '5 / 95 / 1': [29739.843038957268, 0.3384146341463415, 0.45571407938535885, 1.7729377432334783], '10 / 5 / 1': [27560.281893917247, 0.375, 0.9420420649816599, 1.9943067243898], '10 / 10 / 1': [19177.97286379451, 0.33116883116883117, 0.6060227700846127, 1.5870128789830302], '10 / 15 / 1': [17410.463541697332, 0.30864197530864196, 0.5215309538147882, 1.4991981235551421], '10 / 20 / 1': [17355.35234737387, 0.2962962962962963, 0.5199340974887953, 1.4944995947529394], '10 / 25 / 1': [17259.245211143647, 0.3048780487804878, 0.5109567005136575, 1.4829340000330158], '10 / 30 / 1': [19223.326441918547, 0.2976190476190476, 0.5618148224602914, 1.587281113432932], '10 / 35 / 1': [17961.298441759995, 0.28888888888888886, 0.49513613703193643, 1.516399318427425], '10 / 40 / 1': [17736.143584388294, 0.29120879120879123, 0.48480907965691805, 1.4970354744674086], '10 / 45 / 1': [16612.682159026164, 0.2872340425531915, 0.4368908371198823, 1.4510724414640714], '10 / 50 / 1': [16156.465637979903, 0.2894736842105263, 0.4182247420395699, 1.4332599292574615], '10 / 55 / 1': [17010.33376662359, 0.2916666666666667, 0.4414497217580286, 1.4669859942469055], '10 / 60 / 1': [17418.24654931897, 0.2916666666666667, 0.45348612560171525, 1.485905533629707], '10 / 65 / 1': [17234.11146689965, 0.28865979381443296, 0.4440834441732472, 1.4794663728952335], '10 / 70 / 1': [16940.698821989485, 0.2857142857142857, 0.43089710042790913, 1.4687183105945258], '10 / 75 / 1': [15640.039160632154, 0.285, 0.38298620929524824, 1.4132984760192595], '10 / 80 / 1': [15163.897607062236, 0.28217821782178215, 0.3644210378929645, 1.3931994305918602], '10 / 85 / 1': [15681.932799590875, 0.28217821782178215, 0.381382243092092, 1.4121521697463373], '10 / 90 / 1': [15854.959308707272, 0.28217821782178215, 0.3868890561487513, 1.4194468829189424], '10 / 95 / 1': [15791.724227807797, 0.28217821782178215, 0.3848653993605362, 1.4167842343726265], '15 / 5 / 1': [21983.93656722432, 0.3854166666666667, 1.0424673149012018, 1.9258554660480758], '15 / 10 / 1': [22607.877714623024, 0.3627450980392157, 1.010769894604622, 2.0177550866837515], '15 / 15 / 1': [27647.90985206243, 0.3627450980392157, 1.2161100573549595, 2.3710723236508824], '15 / 20 / 1': [23935.72997639222, 0.34545454545454546, 1.0014115829593577, 2.1046869293887314], '15 / 25 / 1': [23117.22944763644, 0.34545454545454546, 0.9698816422063815, 2.040569524411979], '15 / 30 / 1': [22243.90432996115, 0.33620689655172414, 0.8918927727337344, 1.9530129884239025], '15 / 35 / 1': [24430.904624031842, 0.3448275862068966, 0.978349032131791, 2.095850389826462], '15 / 40 / 1': [25666.885527552477, 0.3474576271186441, 1.0048296290253107, 2.156400884950834], '15 / 45 / 1': [25374.57279918883, 0.35, 0.9798556165743257, 2.1350382538136787], '15 / 50 / 1': [25655.703331193145, 0.35, 0.9920761404044591, 2.1088747172336304], '15 / 55 / 1': [22696.800090262917, 0.3359375, 0.841462034072424, 1.8939933576206027], '15 / 60 / 1': [21821.4447453456, 0.3235294117647059, 0.768290909148598, 1.8242235918024288], '15 / 65 / 1': [22800.643291374487, 0.3235294117647059, 0.7996234856169768, 1.8876752138096344], '15 / 70 / 1': [22005.13056736894, 0.32608695652173914, 0.7615677747495523, 1.8461165302506344], '15 / 75 / 1': [21512.6843552966, 0.32857142857142857, 0.7361028783142446, 1.8049657074461383], '15 / 80 / 1': [21512.6843552966, 0.32857142857142857, 0.7361028783142446, 1.8049657074461383], '15 / 85 / 1': [22564.999192960695, 0.3402777777777778, 0.7519524780094876, 1.849602994682731], '15 / 90 / 1': [22131.843981496815, 0.3356164383561644, 0.7294302414449851, 1.8268699662135017], '15 / 95 / 1': [24075.073290777564, 0.3424657534246575, 0.7854114361690233, 1.9340011501065606], '20 / 5 / 1': [18676.544609866673, 0.3783783783783784, 1.1361595426775062, 1.837214929681948], '20 / 10 / 1': [20853.191350575213, 0.39473684210526316, 1.2479941368826617, 2.028892795059399], '20 / 15 / 1': [20236.136223133228, 0.3717948717948718, 1.18076810906659, 1.961711605320375], '20 / 20 / 1': [22139.938760154644, 0.3684210526315789, 1.3436372555823177, 2.129538886840786], '20 / 25 / 1': [22463.177090445675, 0.3717948717948718, 1.336501600584736, 2.146754757808834], '20 / 30 / 1': [22137.64073421312, 0.358974358974359, 1.323819049451932, 2.135665982603634], '20 / 35 / 1': [18842.481539120145, 0.3125, 1.0884838650703035, 1.8814073090764003], '20 / 40 / 1': [19783.252671760794, 0.35365853658536583, 1.1229722836291403, 1.9713179058615238], '20 / 45 / 1': [20591.023254413107, 0.35365853658536583, 1.1744534128725144, 2.0278013249799605], '20 / 50 / 1': [20483.106995463066, 0.35714285714285715, 1.1444494431752863, 1.9865770001559673], '20 / 55 / 1': [19982.772974083095, 0.3409090909090909, 1.0686270098107455, 1.927099553333831], '20 / 60 / 1': [20302.458575045577, 0.32978723404255317, 1.0257127533861503, 1.9283657711226272], '20 / 65 / 1': [19428.57186129123, 0.3229166666666667, 0.944073930115043, 1.8801815307824314], '20 / 70 / 1': [19428.57186129123, 0.3229166666666667, 0.944073930115043, 1.8801815307824314], '20 / 75 / 1': [19839.438650517393, 0.3229166666666667, 0.9669227038073805, 1.9014839605641458], '20 / 80 / 1': [20185.85196822939, 0.3333333333333333, 0.9851222250936158, 1.91857108012413], '20 / 85 / 1': [19810.83893106325, 0.3333333333333333, 0.9656335390095646, 1.892285566197595], '20 / 90 / 1': [19541.351423854412, 0.3431372549019608, 0.8943330647979737, 1.8446008146425046], '20 / 95 / 1': [19541.351423854412, 0.3431372549019608, 0.8943330647979737, 1.8446008146425046], '25 / 5 / 1': [16485.660332585692, 0.34375, 1.0924277361242445, 1.7381623698447195], '25 / 10 / 1': [16486.98279119157, 0.3235294117647059, 1.033625521864788, 1.7374304710827413], '25 / 15 / 1': [16686.411935902644, 0.3235294117647059, 1.050796076221248, 1.7589782565230225], '25 / 20 / 1': [16720.29489156237, 0.3181818181818182, 1.0929304999893872, 1.7749789060930043], '25 / 25 / 1': [18051.644203242773, 0.3181818181818182, 1.2090431703245295, 1.9089791386546593], '25 / 30 / 1': [18093.002309611064, 0.3333333333333333, 1.212730055633242, 1.9150093012148341], '25 / 35 / 1': [16012.4895907359, 0.3, 0.9772995638389675, 1.6845244828043509], '25 / 40 / 1': [14270.411012390205, 0.2972972972972973, 0.7710618504342629, 1.5243822334097719], '25 / 45 / 1': [15101.632536068066, 0.28378378378378377, 0.8408530463886071, 1.5958300323310026], '25 / 50 / 1': [15167.757015950967, 0.2894736842105263, 0.8302723460409984, 1.5934297428335567], '25 / 55 / 1': [16496.830580242087, 0.3026315789473684, 0.9382022782064277, 1.7187451563352232], '25 / 60 / 1': [14776.656272828453, 0.2948717948717949, 0.7722110976996327, 1.5657039032165174], '25 / 65 / 1': [14776.656272828453, 0.2948717948717949, 0.7722110976996327, 1.5657039032165174], '25 / 70 / 1': [15087.902671892885, 0.2948717948717949, 0.8002787119207248, 1.5862655850754637], '25 / 75 / 1': [15406.092627753846, 0.3125, 0.8109897142074841, 1.603819662211614], '25 / 80 / 1': [15786.51026621143, 0.325, 0.8411462789573498, 1.6387537227923683], '25 / 85 / 1': [15083.45609196925, 0.32926829268292684, 0.7599556174182638, 1.5737102266868421], '25 / 90 / 1': [15897.071435375534, 0.34523809523809523, 0.8053064740424251, 1.6417148509406279], '25 / 95 / 1': [15611.295283653059, 0.34523809523809523, 0.7832365898757407, 1.6188045642528004], '30 / 5 / 1': [20283.61266951107, 0.4166666666666667, 1.9195101985105236, 2.158783011473726], '30 / 10 / 1': [19524.257784109115, 0.3958333333333333, 1.8407597307774808, 2.0615697436810034], '30 / 15 / 1': [19890.407184285894, 0.3958333333333333, 1.8789401932093555, 2.107984931621045], '30 / 20 / 1': [20651.42378487648, 0.4166666666666667, 1.9564912649517947, 2.2039888804291654], '30 / 25 / 1': [20923.55993678854, 0.43478260869565216, 2.0653234707485417, 2.2469589250996793], '30 / 30 / 1': [16844.100781038658, 0.36538461538461536, 1.4238253124950853, 1.7668063782649845], '30 / 35 / 1': [16217.56770926796, 0.32142857142857145, 1.25696095727992, 1.7139171145437369], '30 / 40 / 1': [16383.62784223325, 0.32142857142857145, 1.275787037423121, 1.7283720233907387], '30 / 45 / 1': [19199.889509916207, 0.3448275862068966, 1.4975848911924314, 1.9794985678741797], '30 / 50 / 1': [16559.74050086331, 0.3225806451612903, 1.1580776014734036, 1.7361135349162424], '30 / 55 / 1': [16559.74050086331, 0.3225806451612903, 1.1580776014734036, 1.7361135349162424], '30 / 60 / 1': [16376.134447525195, 0.3064516129032258, 1.140370987339301, 1.717364192244095], '30 / 65 / 1': [16376.134447525195, 0.3064516129032258, 1.140370987339301, 1.717364192244095], '30 / 70 / 1': [15869.8275280534, 0.296875, 1.0585310617558417, 1.6673400891916674], '30 / 75 / 1': [16207.38215054193, 0.296875, 1.090975909734714, 1.702156896246106], '30 / 80 / 1': [16207.38215054193, 0.296875, 1.090975909734714, 1.702156896246106], '30 / 85 / 1': [16207.38215054193, 0.296875, 1.090975909734714, 1.702156896246106], '30 / 90 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '30 / 95 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '35 / 5 / 1': [20324.38421526823, 0.39473684210526316, 2.3681367508857494, 2.2997127132564628], '35 / 10 / 1': [18771.31664693003, 0.39473684210526316, 2.148906349193749, 2.1182512208602615], '35 / 15 / 1': [18068.205293840285, 0.375, 1.9506216872856956, 2.017845118949168], '35 / 20 / 1': [18804.333669856507, 0.375, 2.048083796605488, 2.125031079765024], '35 / 25 / 1': [18194.817812751477, 0.35714285714285715, 1.8730682183510845, 2.034120188063226], '35 / 30 / 1': [18744.738399950427, 0.3409090909090909, 1.8595352428609175, 2.109633398999847], '35 / 35 / 1': [18744.738399950427, 0.3409090909090909, 1.8595352428609175, 2.109633398999847], '35 / 40 / 1': [19457.35294423738, 0.34782608695652173, 1.858351587641555, 2.174189895038279], '35 / 45 / 1': [18453.626111224432, 0.34, 1.6107779785932206, 2.028574677294124], '35 / 50 / 1': [15421.17976570274, 0.3148148148148148, 1.158531980193558, 1.6902426972830016], '35 / 55 / 1': [15421.17976570274, 0.3148148148148148, 1.158531980193558, 1.6902426972830016], '35 / 60 / 1': [15421.17976570274, 0.3148148148148148, 1.158531980193558, 1.6902426972830016], '35 / 65 / 1': [16122.282696980794, 0.3148148148148148, 1.2422795779064304, 1.7576596725046176], '35 / 70 / 1': [16122.282696980794, 0.3148148148148148, 1.2422795779064304, 1.7576596725046176], '35 / 75 / 1': [16122.282696980794, 0.3148148148148148, 1.2422795779064304, 1.7576596725046176], '35 / 80 / 1': [16440.232177554502, 0.3148148148148148, 1.2778372203686443, 1.7966219638481111], '35 / 85 / 1': [16440.232177554502, 0.3148148148148148, 1.2778372203686443, 1.7966219638481111], '35 / 90 / 1': [17814.552364591884, 0.3333333333333333, 1.4240552619869957, 1.9520985193826343], '35 / 95 / 1': [17814.552364591884, 0.3333333333333333, 1.4240552619869957, 1.9520985193826343], '40 / 5 / 1': [14641.48995796806, 0.39473684210526316, 1.3991284599371656, 1.673521069592373], '40 / 10 / 1': [14297.248220252977, 0.35714285714285715, 1.2162569021181224, 1.6304742814979925], '40 / 15 / 1': [13936.703584022433, 0.32608695652173914, 1.0598978212098347, 1.5781954015337774], '40 / 20 / 1': [13936.703584022433, 0.32608695652173914, 1.0598978212098347, 1.5781954015337774], '40 / 25 / 1': [14149.632970726621, 0.3125, 1.051885868738573, 1.6073150168520218], '40 / 30 / 1': [15207.869287247682, 0.3125, 1.1995880889009347, 1.7419198868667602], '40 / 35 / 1': [14924.795104461014, 0.25925925925925924, 1.0500887585139866, 1.6915172116389992], '40 / 40 / 1': [14460.490139682764, 0.25, 0.9595692567094145, 1.632413979263308], '40 / 45 / 1': [13276.151230470963, 0.2413793103448276, 0.7816920107638102, 1.515097591155666], '40 / 50 / 1': [12928.333705435738, 0.23333333333333334, 0.7126835014875998, 1.4764837938017519], '40 / 55 / 1': [13353.152969673036, 0.25, 0.7657994153195394, 1.5243303448340522], '40 / 60 / 1': [13575.856446126802, 0.26666666666666666, 0.7938226793016719, 1.54662469901109], '40 / 65 / 1': [13575.856446126802, 0.26666666666666666, 0.7938226793016719, 1.54662469901109], '40 / 70 / 1': [13575.856446126802, 0.26666666666666666, 0.7938226793016719, 1.54662469901109], '40 / 75 / 1': [13789.392639817002, 0.26666666666666666, 0.8194210429415416, 1.574376199425076], '40 / 80 / 1': [13789.392639817002, 0.26666666666666666, 0.8194210429415416, 1.574376199425076], '40 / 85 / 1': [13438.899859147936, 0.26666666666666666, 0.7703959978194656, 1.5388583432234475], '40 / 90 / 1': [13438.899859147936, 0.26666666666666666, 0.7703959978194656, 1.5388583432234475], '40 / 95 / 1': [13310.27864134119, 0.25806451612903225, 0.7324818726554561, 1.524465400142172], '45 / 5 / 1': [9686.233333781396, 0.21739130434782608, 0.265705661631552, 1.115344598188695], '45 / 10 / 1': [10084.576519076245, 0.21739130434782608, 0.35344418878244627, 1.15794958693617], '45 / 15 / 1': [9946.22772599731, 0.22727272727272727, 0.3318322073296574, 1.1415227794031475], '45 / 20 / 1': [10050.548016428991, 0.21739130434782608, 0.3441303113120511, 1.1545707073508698], '45 / 25 / 1': [10704.959363029273, 0.2391304347826087, 0.479381130547635, 1.2268800011927332], '45 / 30 / 1': [11275.903079054613, 0.22916666666666666, 0.5783920655928417, 1.2940470792342558], '45 / 35 / 1': [11447.041327868383, 0.22916666666666666, 0.6090544155286615, 1.3145385583830362], '45 / 40 / 1': [11447.041327868383, 0.22916666666666666, 0.6090544155286615, 1.3145385583830362], '45 / 45 / 1': [9571.241445558147, 0.20833333333333334, 0.22159665565287834, 1.1021255638029113], '45 / 50 / 1': [9581.483631271483, 0.20833333333333334, 0.2237247195394473, 1.1032075286333545], '45 / 55 / 1': [9337.544260464014, 0.21153846153846154, 0.16617807582163072, 1.0801381285497942], '45 / 60 / 1': [9087.655196492256, 0.21153846153846154, 0.11557494020285429, 1.0544074427409116], '45 / 65 / 1': [9814.207595067419, 0.21153846153846154, 0.26310149199286514, 1.128320038127977], '45 / 70 / 1': [9733.49713606174, 0.2037037037037037, 0.24078415775879258, 1.121180496672797], '45 / 75 / 1': [9914.047990713903, 0.2037037037037037, 0.27409530133184123, 1.1402971739722307], '45 / 80 / 1': [9914.047990713903, 0.2037037037037037, 0.27409530133184123, 1.1402971739722307], '45 / 85 / 1': [9914.047990713903, 0.2037037037037037, 0.27409530133184123, 1.1402971739722307], '45 / 90 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 95 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '50 / 5 / 1': [10410.80267651898, 0.25, 0.48991711749408606, 1.1993472036072121], '50 / 10 / 1': [10128.270670305483, 0.22727272727272727, 0.38969158999551473, 1.1701802189519208], '50 / 15 / 1': [10128.270670305483, 0.22727272727272727, 0.38969158999551473, 1.1701802189519208], '50 / 20 / 1': [10199.352719642016, 0.22727272727272727, 0.404707608517757, 1.1791604048539817], '50 / 25 / 1': [10137.278460797042, 0.22727272727272727, 0.39118572437849386, 1.1720593081535309], '50 / 30 / 1': [9697.265474362732, 0.22727272727272727, 0.2869503978484387, 1.1232243219214229], '50 / 35 / 1': [9484.569339392432, 0.21739130434782608, 0.23001210754059798, 1.1012422592675575], '50 / 40 / 1': [9882.419650501215, 0.21739130434782608, 0.3170238402997041, 1.1450985411352983], '50 / 45 / 1': [9464.454345564047, 0.21739130434782608, 0.1965176239142046, 1.0895318212324332], '50 / 50 / 1': [10189.473382955115, 0.21739130434782608, 0.36534475988570964, 1.1727594731080286], '50 / 55 / 1': [9214.044420163471, 0.19230769230769232, 0.13934248501076257, 1.0685449983090796], '50 / 60 / 1': [9796.466553060216, 0.18518518518518517, 0.25009898505836636, 1.1309001397082192], '50 / 65 / 1': [9682.961966827472, 0.19642857142857142, 0.2252406841906353, 1.1185769015502551], '50 / 70 / 1': [9684.488353952895, 0.19642857142857142, 0.22491642383758456, 1.1186211131544583], '50 / 75 / 1': [9463.669827975229, 0.1896551724137931, 0.17960928721670094, 1.0965693790365545], '50 / 80 / 1': [9927.381521931044, 0.1896551724137931, 0.25924080700654994, 1.145619024302963], '50 / 85 / 1': [9927.381521931044, 0.1896551724137931, 0.25924080700654994, 1.145619024302963], '50 / 90 / 1': [9509.642198032418, 0.2, 0.17867067047551807, 1.1012231173788163], '50 / 95 / 1': [10549.48215071142, 0.20689655172413793, 0.3551674096244623, 1.2176200984217964], '55 / 5 / 1': [10082.067476496864, 0.23684210526315788, 0.392236066816368, 1.1556025603172426], '55 / 10 / 1': [10350.173723662985, 0.25, 0.44099975877133246, 1.1869193086428138], '55 / 15 / 1': [10591.742849084672, 0.25, 0.49786015804138717, 1.2162310628533661], '55 / 20 / 1': [10377.678144154734, 0.23809523809523808, 0.42907317317491306, 1.1929002284411079], '55 / 25 / 1': [10639.44317659973, 0.23809523809523808, 0.48581144722013503, 1.2241252949884251], '55 / 30 / 1': [10420.10269138037, 0.22727272727272727, 0.41850412337406623, 1.1989877616493159], '55 / 35 / 1': [10461.278538782173, 0.1875, 0.3908305255352842, 1.2051716152904857], '55 / 40 / 1': [10296.36386519561, 0.1875, 0.35626919467166257, 1.185420505429163], '55 / 45 / 1': [10663.422356848838, 0.1875, 0.43111210483443013, 1.2287188829075582], '55 / 50 / 1': [9963.387251294582, 0.18, 0.28349864948475845, 1.1461429309087223], '55 / 55 / 1': [9889.368798969004, 0.17307692307692307, 0.2603114007234343, 1.138644526765011], '55 / 60 / 1': [10579.186041721061, 0.19230769230769232, 0.3889436522330704, 1.2155313052930707], '55 / 65 / 1': [10019.098668796894, 0.1724137931034483, 0.2652289259347284, 1.1534744634018028], '55 / 70 / 1': [9757.13008923366, 0.16666666666666666, 0.21453086930633986, 1.1256185573488655], '55 / 75 / 1': [9493.184355810432, 0.16129032258064516, 0.1658558414732984, 1.0983345716782493], '55 / 80 / 1': [9493.184355810432, 0.16129032258064516, 0.1658558414732984, 1.0983345716782493], '55 / 85 / 1': [9493.184355810432, 0.16129032258064516, 0.1658558414732984, 1.0983345716782493], '55 / 90 / 1': [9493.184355810432, 0.16129032258064516, 0.1658558414732984, 1.0983345716782493], '55 / 95 / 1': [9493.184355810432, 0.16129032258064516, 0.1658558414732984, 1.0983345716782493], '60 / 5 / 1': [10684.060939061188, 0.3055555555555556, 0.5928891992458195, 1.2265835203689468], '60 / 10 / 1': [10761.297139193339, 0.2894736842105263, 0.5839000826323496, 1.2372066281773693], '60 / 15 / 1': [10543.805652703571, 0.275, 0.5078589920607139, 1.2129088721812082], '60 / 20 / 1': [11422.805543655706, 0.275, 0.6976808526494691, 1.3153091676004962], '60 / 25 / 1': [11310.81353235087, 0.2619047619047619, 0.6445118212331165, 1.3040127546126339], '60 / 30 / 1': [10185.542207552939, 0.2727272727272727, 0.3776666948077549, 1.1707677400812373], '60 / 35 / 1': [11358.644180277133, 0.2727272727272727, 0.6195900183698481, 1.3030377028077786], '60 / 40 / 1': [11509.222697415906, 0.2727272727272727, 0.651056318441265, 1.3207728822584894], '60 / 45 / 1': [11374.097587750079, 0.2608695652173913, 0.6003835699611466, 1.305931337614655], '60 / 50 / 1': [11165.680555273568, 0.22916666666666666, 0.5400161840598785, 1.2831114800134702], '60 / 55 / 1': [11742.187073873438, 0.22916666666666666, 0.6450028881969517, 1.347709533454144], '60 / 60 / 1': [11398.358585719392, 0.22, 0.5632498668264193, 1.3066544349261355], '60 / 65 / 1': [11073.800884662043, 0.21153846153846154, 0.4920768347133736, 1.2667330248878743], '60 / 70 / 1': [10644.566680317366, 0.19642857142857142, 0.3899479436835528, 1.2214842480126842], '60 / 75 / 1': [10657.74819601097, 0.19642857142857142, 0.41165556302312584, 1.2392126906307366], '60 / 80 / 1': [10657.74819601097, 0.19642857142857142, 0.41165556302312584, 1.2392126906307366], '60 / 85 / 1': [10657.74819601097, 0.19642857142857142, 0.41165556302312584, 1.2392126906307366], '60 / 90 / 1': [10657.74819601097, 0.19642857142857142, 0.41165556302312584, 1.2392126906307366], '60 / 95 / 1': [11271.015142774982, 0.21428571428571427, 0.5137722046218481, 1.3094143526940616], '65 / 5 / 1': [14531.474151839773, 0.3333333333333333, 1.7639847879562482, 1.7360476421877644], '65 / 10 / 1': [14345.473360565422, 0.29411764705882354, 1.5357546495418548, 1.6992112284105747], '65 / 15 / 1': [14345.473360565422, 0.29411764705882354, 1.5357546495418548, 1.6992112284105747], '65 / 20 / 1': [14003.836362981214, 0.29411764705882354, 1.4617954202628773, 1.654989165755439], '65 / 25 / 1': [14003.836362981214, 0.29411764705882354, 1.4617954202628773, 1.654989165755439], '65 / 30 / 1': [14003.836362981214, 0.29411764705882354, 1.4617954202628773, 1.654989165755439], '65 / 35 / 1': [13924.052391409405, 0.29411764705882354, 1.4455856327382206, 1.6427154887047648], '65 / 40 / 1': [14834.294332728174, 0.3125, 1.7422113543708118, 1.766076583823685], '65 / 45 / 1': [14584.339123468104, 0.29411764705882354, 1.5914262307422815, 1.7383619860300996], '65 / 50 / 1': [15355.288904006627, 0.29411764705882354, 1.7412757173388238, 1.8372734370118446], '65 / 55 / 1': [15355.288904006627, 0.29411764705882354, 1.7412757173388238, 1.8372734370118446], '65 / 60 / 1': [14199.036991406005, 0.2631578947368421, 1.3701908450540063, 1.6598237126195556], '65 / 65 / 1': [14127.169910858951, 0.275, 1.293319069671974, 1.6475439689699363], '65 / 70 / 1': [13798.563163286206, 0.2619047619047619, 1.1811932950738209, 1.598903329761995], '65 / 75 / 1': [13663.526001170922, 0.2727272727272727, 1.1071679159599705, 1.5851583686742339], '65 / 80 / 1': [12797.879232789586, 0.2708333333333333, 0.8812876890266623, 1.4867091077639922], '65 / 85 / 1': [12797.879232789586, 0.2708333333333333, 0.8812876890266623, 1.4867091077639922], '65 / 90 / 1': [12698.422135569535, 0.28, 0.8311563471935709, 1.4803433985732328], '65 / 95 / 1': [12698.422135569535, 0.28, 0.8311563471935709, 1.4803433985732328], '70 / 5 / 1': [10341.730958045895, 0.29411764705882354, 0.5362174344107763, 1.1956375075343875], '70 / 10 / 1': [10341.730958045895, 0.29411764705882354, 0.5362174344107763, 1.1956375075343875], '70 / 15 / 1': [10503.826961152678, 0.29411764705882354, 0.5845203348679338, 1.2153557870388998], '70 / 20 / 1': [11715.241943923971, 0.3125, 0.9445680371152042, 1.36767106115643], '70 / 25 / 1': [11696.917556694203, 0.3125, 0.9393830484820894, 1.365305158004709], '70 / 30 / 1': [12793.785148699653, 0.34375, 1.2479265076012125, 1.511876445680566], '70 / 35 / 1': [12793.785148699653, 0.34375, 1.2479265076012125, 1.511876445680566], '70 / 40 / 1': [12793.785148699653, 0.34375, 1.2479265076012125, 1.511876445680566], '70 / 45 / 1': [13093.078070070364, 0.34375, 1.3240201553059519, 1.5512477228619488], '70 / 50 / 1': [12170.310253451813, 0.3235294117647059, 1.0483739426346723, 1.4210910395259486], '70 / 55 / 1': [12170.310253451813, 0.3235294117647059, 1.0483739426346723, 1.4210910395259486], '70 / 60 / 1': [12170.310253451813, 0.3235294117647059, 1.0483739426346723, 1.4210910395259486], '70 / 65 / 1': [12170.310253451813, 0.3235294117647059, 1.0483739426346723, 1.4210910395259486], '70 / 70 / 1': [12271.796027860444, 0.3235294117647059, 1.0722351677613464, 1.4327927291987177], '70 / 75 / 1': [11475.802173800335, 0.2777777777777778, 0.8319559181865411, 1.336024362904973], '70 / 80 / 1': [11475.802173800335, 0.2777777777777778, 0.8319559181865411, 1.336024362904973], '70 / 85 / 1': [11981.187916881696, 0.2777777777777778, 0.9486686430590083, 1.4021201677910131], '70 / 90 / 1': [11981.187916881696, 0.2777777777777778, 0.9486686430590083, 1.4021201677910131], '70 / 95 / 1': [10675.347456616286, 0.2777777777777778, 0.6030202677314318, 1.23531302677804], '75 / 5 / 1': [11357.78926462051, 0.3, 0.9359189450167966, 1.3329762419506752], '75 / 10 / 1': [10011.610061049263, 0.3125, 0.46333036687961027, 1.1575223531989864], '75 / 15 / 1': [10011.610061049263, 0.3125, 0.46333036687961027, 1.1575223531989864], '75 / 20 / 1': [11456.971102631533, 0.3125, 0.8656505242524752, 1.3409358155213826], '75 / 25 / 1': [13235.979066660262, 0.3333333333333333, 1.3822238116187981, 1.6168358256845483], '75 / 30 / 1': [12080.920001001681, 0.3125, 1.02194552022042, 1.4303532264542984], '75 / 35 / 1': [12080.920001001681, 0.3125, 1.02194552022042, 1.4303532264542984], '75 / 40 / 1': [12080.920001001681, 0.3125, 1.02194552022042, 1.4303532264542984], '75 / 45 / 1': [12080.920001001681, 0.3125, 1.02194552022042, 1.4303532264542984], '75 / 50 / 1': [12080.920001001681, 0.3125, 1.02194552022042, 1.4303532264542984], '75 / 55 / 1': [12704.544439626648, 0.3125, 1.1932996067246573, 1.5196978624350517], '75 / 60 / 1': [12782.18779932496, 0.3125, 1.2125226662117472, 1.5280697607149174], '75 / 65 / 1': [12782.18779932496, 0.3125, 1.2125226662117472, 1.5280697607149174], '75 / 70 / 1': [12782.18779932496, 0.3125, 1.2125226662117472, 1.5280697607149174], '75 / 75 / 1': [12301.379322420486, 0.29411764705882354, 1.050723846804492, 1.4289734134302234], '75 / 80 / 1': [12301.379322420486, 0.29411764705882354, 1.050723846804492, 1.4289734134302234], '75 / 85 / 1': [12677.948540947986, 0.29411764705882354, 1.137269771689842, 1.481313619353569], '75 / 90 / 1': [12838.630567337184, 0.29411764705882354, 1.1792088428257874, 1.4990630105914764], '75 / 95 / 1': [12838.630567337184, 0.29411764705882354, 1.1792088428257874, 1.4990630105914764], '80 / 5 / 1': [12363.74644222524, 0.23076923076923078, 1.3897986754971403, 1.5164799842830676], '80 / 10 / 1': [12363.74644222524, 0.23076923076923078, 1.3897986754971403, 1.5164799842830676], '80 / 15 / 1': [11284.804163631889, 0.21428571428571427, 0.9775109950505404, 1.3476566193426522], '80 / 20 / 1': [10867.242070922937, 0.21428571428571427, 0.8488496101306456, 1.288687493828347], '80 / 25 / 1': [10867.242070922937, 0.21428571428571427, 0.8488496101306456, 1.288687493828347], '80 / 30 / 1': [11051.260776933426, 0.25, 0.9072400232235652, 1.3133527763779975], '80 / 35 / 1': [10768.052258051057, 0.23333333333333334, 0.7648983037198338, 1.2747372505151802], '80 / 40 / 1': [10768.052258051057, 0.23333333333333334, 0.7648983037198338, 1.2747372505151802], '80 / 45 / 1': [10768.052258051057, 0.23333333333333334, 0.7648983037198338, 1.2747372505151802], '80 / 50 / 1': [11790.546994643057, 0.26666666666666666, 1.0802035247137936, 1.414939782292472], '80 / 55 / 1': [11790.546994643057, 0.26666666666666666, 1.0802035247137936, 1.414939782292472], '80 / 60 / 1': [11790.546994643057, 0.26666666666666666, 1.0802035247137936, 1.414939782292472], '80 / 65 / 1': [11790.546994643057, 0.26666666666666666, 1.0802035247137936, 1.414939782292472], '80 / 70 / 1': [11577.878046814205, 0.26666666666666666, 1.0650165553982986, 1.3569090938085113], '80 / 75 / 1': [12503.313805767264, 0.2857142857142857, 1.405957248361355, 1.479476725734821], '80 / 80 / 1': [12503.313805767264, 0.2857142857142857, 1.405957248361355, 1.479476725734821], '80 / 85 / 1': [13305.999500280048, 0.2857142857142857, 1.619589319540689, 1.5957346435949813], '80 / 90 / 1': [13749.786090150335, 0.2857142857142857, 1.750814922394046, 1.6440034465582856], '80 / 95 / 1': [12298.780405763478, 0.3, 1.2557202300646053, 1.4615624774270117], '85 / 5 / 1': [9609.119456760427, 0.2857142857142857, 0.43900476617810175, 1.1288384141720236], '85 / 10 / 1': [9609.119456760427, 0.2857142857142857, 0.43900476617810175, 1.1288384141720236], '85 / 15 / 1': [10315.172157792504, 0.2857142857142857, 0.6789533615646344, 1.2162411410606273], '85 / 20 / 1': [10315.172157792504, 0.2857142857142857, 0.6789533615646344, 1.2162411410606273], '85 / 25 / 1': [10315.172157792504, 0.2857142857142857, 0.6789533615646344, 1.2162411410606273], '85 / 30 / 1': [10045.3540230364, 0.26666666666666666, 0.5509720634277113, 1.182853089367518], '85 / 35 / 1': [10045.3540230364, 0.26666666666666666, 0.5509720634277113, 1.182853089367518], '85 / 40 / 1': [10589.45323719118, 0.26666666666666666, 0.7532107697876614, 1.2575109456818157], '85 / 45 / 1': [11076.657312947016, 0.3, 0.9024352719936469, 1.32008652137137], '85 / 50 / 1': [10677.949757187085, 0.28125, 0.7963152705409207, 1.2584247363071963], '85 / 55 / 1': [10677.949757187085, 0.28125, 0.7963152705409207, 1.2584247363071963], '85 / 60 / 1': [10677.949757187085, 0.28125, 0.7963152705409207, 1.2584247363071963], '85 / 65 / 1': [10677.949757187085, 0.28125, 0.7963152705409207, 1.2584247363071963], '85 / 70 / 1': [11475.042385223309, 0.28125, 0.9952810519698827, 1.3452894015698904], '85 / 75 / 1': [11843.489035304627, 0.3, 1.144753476639892, 1.3972290365760933], '85 / 80 / 1': [10787.22692024767, 0.3, 0.870869078961439, 1.2759642224322956], '85 / 85 / 1': [10616.933140092493, 0.28125, 0.7715291623079457, 1.2568844652965512], '85 / 90 / 1': [9498.669238066352, 0.29411764705882354, 0.3896562220728936, 1.131850732798553], '85 / 95 / 1': [10091.70496979125, 0.29411764705882354, 0.5900014871100543, 1.1996429776325863], '90 / 5 / 1': [11079.638987940227, 0.3076923076923077, 0.9394390980984902, 1.3336697630413914], '90 / 10 / 1': [11079.638987940227, 0.3076923076923077, 0.9394390980984902, 1.3336697630413914], '90 / 15 / 1': [11854.909054964566, 0.3333333333333333, 1.290662591538046, 1.4563715226820655], '90 / 20 / 1': [11854.909054964566, 0.3333333333333333, 1.290662591538046, 1.4563715226820655], '90 / 25 / 1': [11767.978839438765, 0.3076923076923077, 1.1670523392929657, 1.4429244699031931], '90 / 30 / 1': [13326.137771769074, 0.3076923076923077, 1.6632943300822554, 1.6937841358229435], '90 / 35 / 1': [14011.66091432441, 0.3333333333333333, 2.003759182047429, 1.8165989086831187], '90 / 40 / 1': [13876.390811898775, 0.32142857142857145, 1.745434307257465, 1.6942415496610432], '90 / 45 / 1': [13876.390811898775, 0.32142857142857145, 1.745434307257465, 1.6942415496610432], '90 / 50 / 1': [13876.390811898775, 0.32142857142857145, 1.745434307257465, 1.6942415496610432], '90 / 55 / 1': [14871.830586854136, 0.32142857142857145, 2.00475554132704, 1.8474115812785712], '90 / 60 / 1': [14871.830586854136, 0.32142857142857145, 2.00475554132704, 1.8474115812785712], '90 / 65 / 1': [14871.830586854136, 0.32142857142857145, 2.00475554132704, 1.8474115812785712], '90 / 70 / 1': [14871.830586854136, 0.32142857142857145, 2.00475554132704, 1.8474115812785712], '90 / 75 / 1': [14759.095395784108, 0.3, 1.85019980227759, 1.8300845088248476], '90 / 80 / 1': [14759.095395784108, 0.3, 1.85019980227759, 1.8300845088248476], '90 / 85 / 1': [14759.095395784108, 0.3, 1.85019980227759, 1.8300845088248476], '90 / 90 / 1': [13686.921696526857, 0.34375, 1.4852464626654802, 1.7055096670483478], '90 / 95 / 1': [14700.485496993495, 0.34375, 1.7343450224343495, 1.8238344342720756], '95 / 5 / 1': [10937.624045944014, 0.34615384615384615, 0.9643846482738015, 1.3060088246993917], '95 / 10 / 1': [10937.624045944014, 0.34615384615384615, 0.9643846482738015, 1.3060088246993917], '95 / 15 / 1': [12100.714522713191, 0.34615384615384615, 1.3365630584646218, 1.4514749489817569], '95 / 20 / 1': [14653.687499763266, 0.3333333333333333, 2.2787385948037024, 1.8170580606856483], '95 / 25 / 1': [14653.687499763266, 0.3333333333333333, 2.2787385948037024, 1.8170580606856483], '95 / 30 / 1': [14052.350654064194, 0.3076923076923077, 1.96749628855564, 1.6900925171333683], '95 / 35 / 1': [15193.445314857216, 0.3333333333333333, 2.4494213421675677, 1.8620762388118681], '95 / 40 / 1': [15271.296822770966, 0.3333333333333333, 2.4735817661242607, 1.8738479716340242], '95 / 45 / 1': [18608.193351211565, 0.36363636363636365, 3.603825674538524, 2.5180403993783353], '95 / 50 / 1': [18608.193351211565, 0.36363636363636365, 3.603825674538524, 2.5180403993783353], '95 / 55 / 1': [18608.193351211565, 0.36363636363636365, 3.603825674538524, 2.5180403993783353], '95 / 60 / 1': [20001.22734298445, 0.35, 4.377471978316708, 2.797918626713594], '95 / 65 / 1': [20001.22734298445, 0.35, 4.377471978316708, 2.797918626713594], '95 / 70 / 1': [20001.22734298445, 0.35, 4.377471978316708, 2.797918626713594], '95 / 75 / 1': [20001.22734298445, 0.35, 4.377471978316708, 2.797918626713594], '95 / 80 / 1': [18823.914849319568, 0.3181818181818182, 3.6926713012930468, 2.528322979540751], '95 / 85 / 1': [18823.914849319568, 0.3181818181818182, 3.6926713012930468, 2.528322979540751], '95 / 90 / 1': [18823.914849319568, 0.3181818181818182, 3.6926713012930468, 2.528322979540751], '95 / 95 / 1': [18823.914849319568, 0.3181818181818182, 3.6926713012930468, 2.528322979540751]}
# brute_force_search_results_2 = {'5 / 100 / 1': [29739.843038957268, 0.3384146341463415, 0.45571407938535885, 1.7729377432334783], '5 / 110 / 1': [29954.670712189472, 0.34146341463414637, 0.4579146580775329, 1.7777521182135845], '5 / 120 / 1': [28954.327973480595, 0.3384146341463415, 0.44737729862018966, 1.7536389478198413], '5 / 130 / 1': [27844.173604200387, 0.3373493975903614, 0.43105636428302835, 1.7273566747042277], '5 / 140 / 1': [24762.01265391814, 0.33035714285714285, 0.3923970066259029, 1.6340490958852218], '5 / 150 / 1': [24762.01265391814, 0.33035714285714285, 0.3923970066259029, 1.6340490958852218], '5 / 160 / 1': [25408.124529867102, 0.32840236686390534, 0.3983928919362996, 1.6504114073672183], '5 / 170 / 1': [25408.124529867102, 0.32840236686390534, 0.3983928919362996, 1.6504114073672183], '5 / 180 / 1': [24426.33287598762, 0.32840236686390534, 0.3858442257478128, 1.6256352934730622], '5 / 190 / 1': [24112.754353064818, 0.32941176470588235, 0.38045297432390335, 1.6135578430405007], '5 / 200 / 1': [26178.334099360338, 0.3333333333333333, 0.40328716139778953, 1.652426316149724], '5 / 210 / 1': [25594.570452986383, 0.3313953488372093, 0.3948383832294844, 1.6377762854064548], '5 / 220 / 1': [25594.570452986383, 0.3313953488372093, 0.3948383832294844, 1.6377762854064548], '5 / 230 / 1': [25594.570452986383, 0.3313953488372093, 0.3948383832294844, 1.6377762854064548], '5 / 240 / 1': [25594.570452986383, 0.3313953488372093, 0.3948383832294844, 1.6377762854064548], '5 / 250 / 1': [25594.570452986383, 0.3313953488372093, 0.3948383832294844, 1.6377762854064548], '5 / 260 / 1': [25594.570452986383, 0.3313953488372093, 0.3948383832294844, 1.6377762854064548], '5 / 270 / 1': [25241.622903802298, 0.3275862068965517, 0.387177672740816, 1.629462589233698], '5 / 280 / 1': [24561.638629543817, 0.32857142857142857, 0.37705797139529507, 1.6128390607512502], '5 / 290 / 1': [22550.7857906774, 0.3258426966292135, 0.34817428907003306, 1.5590992694580443], '10 / 100 / 1': [15791.724227807797, 0.28217821782178215, 0.3848653993605362, 1.4167842343726265], '10 / 110 / 1': [15616.840999974234, 0.28431372549019607, 0.3761678139600448, 1.4102442424247938], '10 / 120 / 1': [15427.412240174795, 0.2815533980582524, 0.3672532434309548, 1.4030076517535934], '10 / 130 / 1': [15552.93566417284, 0.2815533980582524, 0.371295353191609, 1.4083245102232171], '10 / 140 / 1': [15389.62411943735, 0.27884615384615385, 0.36343890164837966, 1.4002140621542516], '10 / 150 / 1': [15389.62411943735, 0.27884615384615385, 0.36343890164837966, 1.4002140621542516], '10 / 160 / 1': [15219.133232485565, 0.2761904761904762, 0.35539414743810055, 1.393156639919522], '10 / 170 / 1': [14498.312912374344, 0.27358490566037735, 0.3295762649784941, 1.3620161683533838], '10 / 180 / 1': [14498.312912374344, 0.27358490566037735, 0.3295762649784941, 1.3620161683533838], '10 / 190 / 1': [13867.732722876146, 0.27314814814814814, 0.3035185509860082, 1.3345483239496376], '10 / 200 / 1': [13057.492907350148, 0.2681818181818182, 0.27227578413898734, 1.2968801428048249], '10 / 210 / 1': [13545.633320708592, 0.2727272727272727, 0.2888714416592156, 1.3179624208465666], '10 / 220 / 1': [13545.633320708592, 0.2727272727272727, 0.2888714416592156, 1.3179624208465666], '10 / 230 / 1': [13545.633320708592, 0.2727272727272727, 0.2888714416592156, 1.3179624208465666], '10 / 240 / 1': [13548.710903069976, 0.2727272727272727, 0.2889879795404849, 1.3181087705922], '10 / 250 / 1': [13548.710903069976, 0.2727272727272727, 0.2889879795404849, 1.3181087705922], '10 / 260 / 1': [13548.710903069976, 0.2727272727272727, 0.2889879795404849, 1.3181087705922], '10 / 270 / 1': [13239.631482175382, 0.2747747747747748, 0.26914800594627214, 1.2973579028790911], '10 / 280 / 1': [12868.410580915552, 0.27232142857142855, 0.254021721537649, 1.280986574460249], '10 / 290 / 1': [12868.410580915552, 0.27232142857142855, 0.254021721537649, 1.280986574460249], '15 / 100 / 1': [25165.681480347903, 0.3541666666666667, 0.8258417665970564, 1.998664185710205], '15 / 110 / 1': [24737.795768831715, 0.33783783783783783, 0.7940510175749096, 1.9774027382721528], '15 / 120 / 1': [23860.19752381738, 0.3310810810810811, 0.7677927829295166, 1.9392049291460574], '15 / 130 / 1': [23247.266677905034, 0.3310810810810811, 0.750066176795924, 1.9077644116206984], '15 / 140 / 1': [21806.6962785392, 0.32894736842105265, 0.6885560916796633, 1.8254429190009487], '15 / 150 / 1': [21784.89025966657, 0.32894736842105265, 0.687902697797519, 1.824014186260685], '15 / 160 / 1': [21784.89025966657, 0.32894736842105265, 0.687902697797519, 1.824014186260685], '15 / 170 / 1': [21493.097715444674, 0.3246753246753247, 0.6719270572207292, 1.7991484955072408], '15 / 180 / 1': [21493.097715444674, 0.3246753246753247, 0.6719270572207292, 1.7991484955072408], '15 / 190 / 1': [20863.13940967172, 0.32051282051282054, 0.6447468328627161, 1.7637763795585355], '15 / 200 / 1': [20652.782621864426, 0.3227848101265823, 0.6304070700998009, 1.7531702666888138], '15 / 210 / 1': [20652.782621864426, 0.3227848101265823, 0.6304070700998009, 1.7531702666888138], '15 / 220 / 1': [20652.782621864426, 0.3227848101265823, 0.6304070700998009, 1.7531702666888138], '15 / 230 / 1': [20652.782621864426, 0.3227848101265823, 0.6304070700998009, 1.7531702666888138], '15 / 240 / 1': [20652.782621864426, 0.3227848101265823, 0.6304070700998009, 1.7531702666888138], '15 / 250 / 1': [20652.782621864426, 0.3227848101265823, 0.6304070700998009, 1.7531702666888138], '15 / 260 / 1': [21268.525376066053, 0.3269230769230769, 0.6559617318882595, 1.7926517327279214], '15 / 270 / 1': [21268.525376066053, 0.3269230769230769, 0.6559617318882595, 1.7926517327279214], '15 / 280 / 1': [21268.525376066053, 0.3269230769230769, 0.6559617318882595, 1.7926517327279214], '15 / 290 / 1': [21268.525376066053, 0.3269230769230769, 0.6559617318882595, 1.7926517327279214], '20 / 100 / 1': [18831.229099625187, 0.3431372549019608, 0.8545325736825419, 1.8070135571179724], '20 / 110 / 1': [18598.50017557066, 0.33653846153846156, 0.826980561885644, 1.791122382165071], '20 / 120 / 1': [18598.50017557066, 0.33653846153846156, 0.826980561885644, 1.791122382165071], '20 / 130 / 1': [18598.50017557066, 0.33653846153846156, 0.826980561885644, 1.791122382165071], '20 / 140 / 1': [18490.211694148817, 0.33962264150943394, 0.8043176796028251, 1.7799787598583678], '20 / 150 / 1': [18081.573320691205, 0.3425925925925926, 0.7696071060206452, 1.7532174627409605], '20 / 160 / 1': [18081.573320691205, 0.3425925925925926, 0.7696071060206452, 1.7532174627409605], '20 / 170 / 1': [17687.90675545187, 0.32727272727272727, 0.7380081942651154, 1.713293581105749], '20 / 180 / 1': [17484.865099775172, 0.32727272727272727, 0.7265147840025615, 1.6986417433989425], '20 / 190 / 1': [17484.865099775172, 0.32727272727272727, 0.7265147840025615, 1.6986417433989425], '20 / 200 / 1': [17707.30653605588, 0.32727272727272727, 0.7389200945888056, 1.7105711191063862], '20 / 210 / 1': [17970.153546141664, 0.32727272727272727, 0.7528205833573718, 1.729029183685639], '20 / 220 / 1': [19639.60275427606, 0.3333333333333333, 0.8476428376986069, 1.8520831419628963], '20 / 230 / 1': [19639.60275427606, 0.3333333333333333, 0.8476428376986069, 1.8520831419628963], '20 / 240 / 1': [19639.60275427606, 0.3333333333333333, 0.8476428376986069, 1.8520831419628963], '20 / 250 / 1': [19639.60275427606, 0.3333333333333333, 0.8476428376986069, 1.8520831419628963], '20 / 260 / 1': [20398.558799202878, 0.33962264150943394, 0.8988222971453885, 1.9082257628968087], '20 / 270 / 1': [20398.558799202878, 0.33962264150943394, 0.8988222971453885, 1.9082257628968087], '20 / 280 / 1': [20398.558799202878, 0.33962264150943394, 0.8988222971453885, 1.9082257628968087], '20 / 290 / 1': [19802.983776762518, 0.3425925925925926, 0.8469949197376869, 1.8604678381385922], '25 / 100 / 1': [15418.360455276183, 0.3372093023255814, 0.7515759240703483, 1.6039079459839682], '25 / 110 / 1': [15418.360455276183, 0.3372093023255814, 0.7515759240703483, 1.6039079459839682], '25 / 120 / 1': [15418.360455276183, 0.3372093023255814, 0.7515759240703483, 1.6039079459839682], '25 / 130 / 1': [14494.202129709229, 0.3333333333333333, 0.7056230303203745, 1.5379936588228813], '25 / 140 / 1': [14494.202129709229, 0.3333333333333333, 0.7056230303203745, 1.5379936588228813], '25 / 150 / 1': [15054.318465073462, 0.34146341463414637, 0.7668869558567443, 1.5899245863255012], '25 / 160 / 1': [15054.318465073462, 0.34146341463414637, 0.7668869558567443, 1.5899245863255012], '25 / 170 / 1': [14624.303888190745, 0.3333333333333333, 0.7202466595430237, 1.5358269980871428], '25 / 180 / 1': [15722.348841192117, 0.34523809523809523, 0.8046569702413778, 1.6334250850106882], '25 / 190 / 1': [15722.348841192117, 0.34523809523809523, 0.8046569702413778, 1.6334250850106882], '25 / 200 / 1': [15041.173230340932, 0.3372093023255814, 0.7378798368035836, 1.56942680466025], '25 / 210 / 1': [14978.205204481235, 0.3333333333333333, 0.7049859083689294, 1.5543474817443073], '25 / 220 / 1': [14978.205204481235, 0.3333333333333333, 0.7049859083689294, 1.5543474817443073], '25 / 230 / 1': [14978.205204481235, 0.3333333333333333, 0.7049859083689294, 1.5543474817443073], '25 / 240 / 1': [14978.205204481235, 0.3333333333333333, 0.7049859083689294, 1.5543474817443073], '25 / 250 / 1': [14978.205204481235, 0.3333333333333333, 0.7049859083689294, 1.5543474817443073], '25 / 260 / 1': [14978.205204481235, 0.3333333333333333, 0.7049859083689294, 1.5543474817443073], '25 / 270 / 1': [14978.205204481235, 0.3333333333333333, 0.7049859083689294, 1.5543474817443073], '25 / 280 / 1': [15131.574858453225, 0.3333333333333333, 0.716764151199499, 1.5661280076165325], '25 / 290 / 1': [14404.481444444438, 0.33695652173913043, 0.6377335330828852, 1.504336270156848], '30 / 100 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '30 / 110 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '30 / 120 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '30 / 130 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '30 / 140 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '30 / 150 / 1': [17224.000042676562, 0.3125, 1.1859649183457368, 1.7913755444366561], '30 / 160 / 1': [17136.68998466058, 0.3125, 1.1777557373691596, 1.7858976884189761], '30 / 170 / 1': [16887.2487538868, 0.3125, 1.1533941082743902, 1.766327480829253], '30 / 180 / 1': [18025.65129888646, 0.328125, 1.2571450977423868, 1.8596438342878263], '30 / 190 / 1': [18025.65129888646, 0.328125, 1.2571450977423868, 1.8596438342878263], '30 / 200 / 1': [17651.4585158184, 0.3181818181818182, 1.1912976336681964, 1.8190699281699318], '30 / 210 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 220 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 230 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 240 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 250 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 260 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 270 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 280 / 1': [17610.36460391313, 0.3194444444444444, 1.0965119299576067, 1.7986213077895088], '30 / 290 / 1': [17457.244956766484, 0.32432432432432434, 1.0470450626521337, 1.7809197187790997], '35 / 100 / 1': [17238.974191558867, 0.3148148148148148, 1.3578704547471816, 1.9029687344789965], '35 / 110 / 1': [17238.974191558867, 0.3148148148148148, 1.3578704547471816, 1.9029687344789965], '35 / 120 / 1': [17238.974191558867, 0.3148148148148148, 1.3578704547471816, 1.9029687344789965], '35 / 130 / 1': [16851.71235034627, 0.3148148148148148, 1.3146228826067305, 1.86245829133588], '35 / 140 / 1': [16851.71235034627, 0.3148148148148148, 1.3146228826067305, 1.86245829133588], '35 / 150 / 1': [16851.71235034627, 0.3148148148148148, 1.3146228826067305, 1.86245829133588], '35 / 160 / 1': [16851.71235034627, 0.3148148148148148, 1.3146228826067305, 1.86245829133588], '35 / 170 / 1': [16851.71235034627, 0.3148148148148148, 1.3146228826067305, 1.86245829133588], '35 / 180 / 1': [17646.27645236532, 0.3333333333333333, 1.4009201491155072, 1.9403420953486088], '35 / 190 / 1': [17646.27645236532, 0.3333333333333333, 1.4009201491155072, 1.9403420953486088], '35 / 200 / 1': [17144.646278949105, 0.32142857142857145, 1.303115819099918, 1.873497849250458], '35 / 210 / 1': [18544.862401704788, 0.3275862068965517, 1.4086127887486606, 2.005055185827375], '35 / 220 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '35 / 230 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '35 / 240 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '35 / 250 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '35 / 260 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '35 / 270 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '35 / 280 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '35 / 290 / 1': [18765.080466157582, 0.3275862068965517, 1.4281970672400097, 2.0264803734650267], '40 / 100 / 1': [13310.27864134119, 0.25806451612903225, 0.7324818726554561, 1.524465400142172], '40 / 110 / 1': [13310.27864134119, 0.25806451612903225, 0.7324818726554561, 1.524465400142172], '40 / 120 / 1': [13640.631308844751, 0.25806451612903225, 0.7733014587200141, 1.5536926907253157], '40 / 130 / 1': [13640.631308844751, 0.25806451612903225, 0.7733014587200141, 1.5536926907253157], '40 / 140 / 1': [13640.631308844751, 0.25806451612903225, 0.7733014587200141, 1.5536926907253157], '40 / 150 / 1': [13640.631308844751, 0.25806451612903225, 0.7733014587200141, 1.5536926907253157], '40 / 160 / 1': [13640.631308844751, 0.25806451612903225, 0.7733014587200141, 1.5536926907253157], '40 / 170 / 1': [13640.631308844751, 0.25806451612903225, 0.7733014587200141, 1.5536926907253157], '40 / 180 / 1': [14283.79182224281, 0.27419354838709675, 0.8484635940663678, 1.620858005046852], '40 / 190 / 1': [14283.79182224281, 0.27419354838709675, 0.8484635940663678, 1.620858005046852], '40 / 200 / 1': [13790.16201044483, 0.265625, 0.7731399676325759, 1.54919539947745], '40 / 210 / 1': [13285.896842645025, 0.25, 0.6778125912425348, 1.4930237730390503], '40 / 220 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '40 / 230 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '40 / 240 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '40 / 250 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '40 / 260 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '40 / 270 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '40 / 280 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '40 / 290 / 1': [15297.56297629294, 0.2647058823529412, 0.8882297184202195, 1.7360508399716672], '45 / 100 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 110 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 120 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 130 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 140 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 150 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 160 / 1': [9643.84433441726, 0.19642857142857142, 0.2147528127361639, 1.1131250554254808], '45 / 170 / 1': [10114.116042012192, 0.19642857142857142, 0.2992793330135727, 1.1612563402292675], '45 / 180 / 1': [10114.116042012192, 0.19642857142857142, 0.2992793330135727, 1.1612563402292675], '45 / 190 / 1': [10843.918799667066, 0.19642857142857142, 0.4187191906543912, 1.2411305971985014], '45 / 200 / 1': [10843.918799667066, 0.19642857142857142, 0.4187191906543912, 1.2411305971985014], '45 / 210 / 1': [10517.219362939755, 0.1896551724137931, 0.35955858991547235, 1.2029226962924042], '45 / 220 / 1': [12081.519133508456, 0.1896551724137931, 0.6021183868380763, 1.3816313703830163], '45 / 230 / 1': [12081.519133508456, 0.1896551724137931, 0.6021183868380763, 1.3816313703830163], '45 / 240 / 1': [11708.851823998215, 0.1896551724137931, 0.5452794800831173, 1.3397962427750265], '45 / 250 / 1': [11708.851823998215, 0.1896551724137931, 0.5452794800831173, 1.3397962427750265], '45 / 260 / 1': [11708.851823998215, 0.1896551724137931, 0.5452794800831173, 1.3397962427750265], '45 / 270 / 1': [11708.851823998215, 0.1896551724137931, 0.5452794800831173, 1.3397962427750265], '45 / 280 / 1': [11708.851823998215, 0.1896551724137931, 0.5452794800831173, 1.3397962427750265], '45 / 290 / 1': [11426.77695051699, 0.2, 0.47715123776089735, 1.3038591255103336], '50 / 100 / 1': [10542.892404490147, 0.21666666666666667, 0.3449543125923318, 1.218135239134983], '50 / 110 / 1': [10945.309647955066, 0.21666666666666667, 0.40543188851243606, 1.2665735559242697], '50 / 120 / 1': [10945.309647955066, 0.21666666666666667, 0.40543188851243606, 1.2665735559242697], '50 / 130 / 1': [10945.309647955066, 0.21666666666666667, 0.40543188851243606, 1.2665735559242697], '50 / 140 / 1': [10945.309647955066, 0.21666666666666667, 0.40543188851243606, 1.2665735559242697], '50 / 150 / 1': [10945.309647955066, 0.21666666666666667, 0.40543188851243606, 1.2665735559242697], '50 / 160 / 1': [10945.309647955066, 0.21666666666666667, 0.40543188851243606, 1.2665735559242697], '50 / 170 / 1': [10347.580843348947, 0.20967741935483872, 0.30935369220311826, 1.1915594737467212], '50 / 180 / 1': [10347.580843348947, 0.20967741935483872, 0.30935369220311826, 1.1915594737467212], '50 / 190 / 1': [11094.229685735605, 0.20967741935483872, 0.4172348539432123, 1.2768569519294937], '50 / 200 / 1': [11094.229685735605, 0.20967741935483872, 0.4172348539432123, 1.2768569519294937], '50 / 210 / 1': [11094.229685735605, 0.20967741935483872, 0.4172348539432123, 1.2768569519294937], '50 / 220 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '50 / 230 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '50 / 240 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '50 / 250 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '50 / 260 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '50 / 270 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '50 / 280 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '50 / 290 / 1': [13029.25389385047, 0.20967741935483872, 0.6871812868554501, 1.5039407772792428], '55 / 100 / 1': [9814.00101391948, 0.1774193548387097, 0.21902370391831102, 1.1335933839761383], '55 / 110 / 1': [8990.196739138844, 0.1875, 0.07726131117740294, 1.0466686602054873], '55 / 120 / 1': [8990.196739138844, 0.1875, 0.07726131117740294, 1.0466686602054873], '55 / 130 / 1': [8990.196739138844, 0.1875, 0.07726131117740294, 1.0466686602054873], '55 / 140 / 1': [9166.436762490044, 0.203125, 0.10745397186015881, 1.0658665167846753], '55 / 150 / 1': [9219.647731188204, 0.203125, 0.11651289366780371, 1.071617918258934], '55 / 160 / 1': [9219.647731188204, 0.203125, 0.11651289366780371, 1.071617918258934], '55 / 170 / 1': [8716.158182285199, 0.19696969696969696, 0.035012739402827384, 1.0204516017223755], '55 / 180 / 1': [8716.158182285199, 0.19696969696969696, 0.035012739402827384, 1.0204516017223755], '55 / 190 / 1': [9345.08870386168, 0.19696969696969696, 0.136355648916249, 1.0846594330281483], '55 / 200 / 1': [9345.08870386168, 0.19696969696969696, 0.136355648916249, 1.0846594330281483], '55 / 210 / 1': [9781.362773713108, 0.19696969696969696, 0.20691512158783448, 1.1314205006949543], '55 / 220 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '55 / 230 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '55 / 240 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '55 / 250 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '55 / 260 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '55 / 270 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '55 / 280 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '55 / 290 / 1': [10332.372427037584, 0.19696969696969696, 0.292809885257465, 1.1915023332147798], '60 / 100 / 1': [10200.488064616333, 0.20689655172413793, 0.3266443329001167, 1.1927920641977687], '60 / 110 / 1': [10200.488064616333, 0.20689655172413793, 0.3266443329001167, 1.1927920641977687], '60 / 120 / 1': [10200.488064616333, 0.20689655172413793, 0.3266443329001167, 1.1927920641977687], '60 / 130 / 1': [9899.179751276773, 0.1896551724137931, 0.2750353685405632, 1.159579841043943], '60 / 140 / 1': [9899.179751276773, 0.1896551724137931, 0.2750353685405632, 1.159579841043943], '60 / 150 / 1': [9899.179751276773, 0.1896551724137931, 0.2750353685405632, 1.159579841043943], '60 / 160 / 1': [9899.179751276773, 0.1896551724137931, 0.2750353685405632, 1.159579841043943], '60 / 170 / 1': [9544.249546422274, 0.18333333333333332, 0.21101513781310585, 1.1187627726227918], '60 / 180 / 1': [9544.249546422274, 0.18333333333333332, 0.21101513781310585, 1.1187627726227918], '60 / 190 / 1': [10232.93253263601, 0.18333333333333332, 0.32320217671451773, 1.1941630661170106], '60 / 200 / 1': [10232.93253263601, 0.18333333333333332, 0.32320217671451773, 1.1941630661170106], '60 / 210 / 1': [9853.487489749057, 0.1935483870967742, 0.23144201978870974, 1.1382973444439195], '60 / 220 / 1': [10403.490873198729, 0.1935483870967742, 0.3225862800645329, 1.1985266109145365], '60 / 230 / 1': [10403.490873198729, 0.1935483870967742, 0.3225862800645329, 1.1985266109145365], '60 / 240 / 1': [10231.92117754732, 0.1875, 0.2890734952779548, 1.1807905262595249], '60 / 250 / 1': [9921.991072038603, 0.1875, 0.2392486665454274, 1.1474170340694787], '60 / 260 / 1': [9921.991072038603, 0.1875, 0.2392486665454274, 1.1474170340694787], '60 / 270 / 1': [9921.991072038603, 0.1875, 0.2392486665454274, 1.1474170340694787], '60 / 280 / 1': [9921.991072038603, 0.1875, 0.2392486665454274, 1.1474170340694787], '60 / 290 / 1': [9921.991072038603, 0.1875, 0.2392486665454274, 1.1474170340694787], '65 / 100 / 1': [11492.319173785258, 0.2692307692307692, 0.610412260632094, 1.345112681510809], '65 / 110 / 1': [11395.412797799383, 0.2692307692307692, 0.5944266002823586, 1.3330645573898954], '65 / 120 / 1': [11395.412797799383, 0.2692307692307692, 0.5944266002823586, 1.3330645573898954], '65 / 130 / 1': [11395.412797799383, 0.2692307692307692, 0.5944266002823586, 1.3330645573898954], '65 / 140 / 1': [11395.412797799383, 0.2692307692307692, 0.5944266002823586, 1.3330645573898954], '65 / 150 / 1': [11395.412797799383, 0.2692307692307692, 0.5944266002823586, 1.3330645573898954], '65 / 160 / 1': [11395.412797799383, 0.2692307692307692, 0.5944266002823586, 1.3330645573898954], '65 / 170 / 1': [10765.7922600197, 0.25, 0.4575802276129516, 1.2576372083175666], '65 / 180 / 1': [10313.812663528053, 0.2413793103448276, 0.37196972569801756, 1.20842697704157], '65 / 190 / 1': [11058.024900416818, 0.2413793103448276, 0.4880252831822368, 1.292476433746537], '65 / 200 / 1': [10711.775350533064, 0.25, 0.39612906263854675, 1.236117244554881], '65 / 210 / 1': [10711.775350533064, 0.25, 0.39612906263854675, 1.236117244554881], '65 / 220 / 1': [10054.493853257645, 0.24193548387096775, 0.27688535373461803, 1.1644990006679752], '65 / 230 / 1': [10054.493853257645, 0.24193548387096775, 0.27688535373461803, 1.1644990006679752], '65 / 240 / 1': [9741.637565694204, 0.234375, 0.22145233324926358, 1.132022435157044], '65 / 250 / 1': [9741.637565694204, 0.234375, 0.22145233324926358, 1.132022435157044], '65 / 260 / 1': [9741.637565694204, 0.234375, 0.22145233324926358, 1.132022435157044], '65 / 270 / 1': [9741.637565694204, 0.234375, 0.22145233324926358, 1.132022435157044], '65 / 280 / 1': [9741.637565694204, 0.234375, 0.22145233324926358, 1.132022435157044], '65 / 290 / 1': [9741.637565694204, 0.234375, 0.22145233324926358, 1.132022435157044], '70 / 100 / 1': [10675.347456616286, 0.2777777777777778, 0.6030202677314318, 1.23531302677804], '70 / 110 / 1': [10675.347456616286, 0.2777777777777778, 0.6030202677314318, 1.23531302677804], '70 / 120 / 1': [10675.347456616286, 0.2777777777777778, 0.6030202677314318, 1.23531302677804], '70 / 130 / 1': [10675.347456616286, 0.2777777777777778, 0.6030202677314318, 1.23531302677804], '70 / 140 / 1': [10405.410412769945, 0.25, 0.5338266005333118, 1.2047539825181766], '70 / 150 / 1': [10405.410412769945, 0.25, 0.5338266005333118, 1.2047539825181766], '70 / 160 / 1': [10405.410412769945, 0.25, 0.5338266005333118, 1.2047539825181766], '70 / 170 / 1': [10111.508711414192, 0.2631578947368421, 0.4363419250410769, 1.171414411307685], '70 / 180 / 1': [10111.508711414192, 0.2631578947368421, 0.4363419250410769, 1.171414411307685], '70 / 190 / 1': [10610.405095803044, 0.275, 0.4955415965279843, 1.2148795482667456], '70 / 200 / 1': [10610.405095803044, 0.275, 0.4955415965279843, 1.2148795482667456], '70 / 210 / 1': [10610.405095803044, 0.275, 0.4955415965279843, 1.2148795482667456], '70 / 220 / 1': [10981.026801499955, 0.2619047619047619, 0.5622742047153788, 1.2592580546239491], '70 / 230 / 1': [10981.026801499955, 0.2619047619047619, 0.5622742047153788, 1.2592580546239491], '70 / 240 / 1': [10648.72605240417, 0.25, 0.4709482680791319, 1.2204845960004778], '70 / 250 / 1': [9741.65686214995, 0.22, 0.24756220095017376, 1.120967102107991], '70 / 260 / 1': [9810.731570829776, 0.22, 0.2621026105108822, 1.1285149876877465], '70 / 270 / 1': [9810.731570829776, 0.22, 0.2621026105108822, 1.1285149876877465], '70 / 280 / 1': [9810.731570829776, 0.22, 0.2621026105108822, 1.1285149876877465], '70 / 290 / 1': [9810.731570829776, 0.22, 0.2621026105108822, 1.1285149876877465], '75 / 100 / 1': [11619.210533320103, 0.3055555555555556, 0.8373342379769254, 1.353605187209396], '75 / 110 / 1': [11678.747593021875, 0.3055555555555556, 0.8518948491991893, 1.3608337055635835], '75 / 120 / 1': [12609.056712455991, 0.3235294117647059, 1.1197712130750637, 1.4906931899442317], '75 / 130 / 1': [12720.169922469186, 0.3235294117647059, 1.145092161545195, 1.5046361993207127], '75 / 140 / 1': [12720.169922469186, 0.3235294117647059, 1.145092161545195, 1.5046361993207127], '75 / 150 / 1': [12720.169922469186, 0.3235294117647059, 1.145092161545195, 1.5046361993207127], '75 / 160 / 1': [12720.169922469186, 0.3235294117647059, 1.145092161545195, 1.5046361993207127], '75 / 170 / 1': [12709.823626420037, 0.2777777777777778, 1.0812212253684268, 1.5094751469044438], '75 / 180 / 1': [12566.172866678367, 0.2631578947368421, 0.9973669814133735, 1.489510689915135], '75 / 190 / 1': [12332.373012572976, 0.275, 0.8780874822015108, 1.4401516285871407], '75 / 200 / 1': [12332.373012572976, 0.275, 0.8780874822015108, 1.4401516285871407], '75 / 210 / 1': [12332.373012572976, 0.275, 0.8780874822015108, 1.4401516285871407], '75 / 220 / 1': [12763.143099100429, 0.2619047619047619, 0.9266036196425467, 1.4948671800679434], '75 / 230 / 1': [12763.143099100429, 0.2619047619047619, 0.9266036196425467, 1.4948671800679434], '75 / 240 / 1': [12269.785027307667, 0.25, 0.7989812751151071, 1.4266187903999075], '75 / 250 / 1': [12478.381754449843, 0.2391304347826087, 0.8004378209264283, 1.4586166282881732], '75 / 260 / 1': [11547.414822348885, 0.22916666666666666, 0.6119502756099838, 1.334811653688544], '75 / 270 / 1': [11547.414822348885, 0.22916666666666666, 0.6119502756099838, 1.334811653688544], '75 / 280 / 1': [11547.414822348885, 0.22916666666666666, 0.6119502756099838, 1.334811653688544], '75 / 290 / 1': [11547.414822348885, 0.22916666666666666, 0.6119502756099838, 1.334811653688544], '80 / 100 / 1': [12298.780405763478, 0.3, 1.2557202300646053, 1.4615624774270117], '80 / 110 / 1': [12361.79959464702, 0.3, 1.2748980698297823, 1.47008064184285], '80 / 120 / 1': [12361.79959464702, 0.3, 1.2748980698297823, 1.47008064184285], '80 / 130 / 1': [12361.79959464702, 0.3, 1.2748980698297823, 1.47008064184285], '80 / 140 / 1': [12361.79959464702, 0.3, 1.2748980698297823, 1.47008064184285], '80 / 150 / 1': [12811.36603665379, 0.2857142857142857, 1.4853721067457413, 1.5421713152638978], '80 / 160 / 1': [13971.410819300123, 0.3076923076923077, 1.9147720056808561, 1.7323675260272935], '80 / 170 / 1': [13469.327400751596, 0.3, 1.545535173901991, 1.6560038091574032], '80 / 180 / 1': [12138.858626783156, 0.2647058823529412, 1.0456981363576563, 1.4426640793138856], '80 / 190 / 1': [12138.858626783156, 0.2647058823529412, 1.0456981363576563, 1.4426640793138856], '80 / 200 / 1': [11652.65149041324, 0.25, 0.8735430926239314, 1.376830584794863], '80 / 210 / 1': [11652.65149041324, 0.25, 0.8735430926239314, 1.376830584794863], '80 / 220 / 1': [11725.74606936423, 0.225, 0.8150013802461993, 1.3837322301203852], '80 / 230 / 1': [11725.74606936423, 0.225, 0.8150013802461993, 1.3837322301203852], '80 / 240 / 1': [11725.74606936423, 0.225, 0.8150013802461993, 1.3837322301203852], '80 / 250 / 1': [11469.6683255726, 0.20454545454545456, 0.6976344209990917, 1.3533978956070885], '80 / 260 / 1': [10824.975353988175, 0.21739130434782608, 0.5470235606292391, 1.2696517935487386], '80 / 270 / 1': [10824.975353988175, 0.21739130434782608, 0.5470235606292391, 1.2696517935487386], '80 / 280 / 1': [10824.975353988175, 0.21739130434782608, 0.5470235606292391, 1.2696517935487386], '80 / 290 / 1': [10824.975353988175, 0.21739130434782608, 0.5470235606292391, 1.2696517935487386], '85 / 100 / 1': [10091.70496979125, 0.29411764705882354, 0.5900014871100543, 1.1996429776325863], '85 / 110 / 1': [10091.70496979125, 0.29411764705882354, 0.5900014871100543, 1.1996429776325863], '85 / 120 / 1': [13310.271876746765, 0.3125, 1.4192306395578582, 1.613239448835714], '85 / 130 / 1': [13310.271876746765, 0.3125, 1.4192306395578582, 1.613239448835714], '85 / 140 / 1': [13794.331784364347, 0.3, 1.6252952453279588, 1.7025115663390575], '85 / 150 / 1': [13794.331784364347, 0.3, 1.6252952453279588, 1.7025115663390575], '85 / 160 / 1': [13735.782186770239, 0.3, 1.6112562537953357, 1.6943170397165475], '85 / 170 / 1': [13735.782186770239, 0.3, 1.6112562537953357, 1.6943170397165475], '85 / 180 / 1': [12660.2997270864, 0.2777777777777778, 1.1069226757784796, 1.5163154049225867], '85 / 190 / 1': [12645.474321324571, 0.2631578947368421, 1.0386790529160215, 1.5191438981269705], '85 / 200 / 1': [12645.474321324571, 0.2631578947368421, 1.0386790529160215, 1.5191438981269705], '85 / 210 / 1': [12645.474321324571, 0.2631578947368421, 1.0386790529160215, 1.5191438981269705], '85 / 220 / 1': [12119.487161839399, 0.275, 0.8753247553030811, 1.4458956724379786], '85 / 230 / 1': [11029.715656122044, 0.22727272727272727, 0.5907189820345068, 1.298940333576487], '85 / 240 / 1': [10987.003855369949, 0.22727272727272727, 0.5802743468426833, 1.2936547022978464], '85 / 250 / 1': [10684.033690649436, 0.22727272727272727, 0.5183693529099453, 1.2543584675582815], '85 / 260 / 1': [10083.158520302464, 0.2391304347826087, 0.3756051538341304, 1.1797436263554544], '85 / 270 / 1': [10083.158520302464, 0.2391304347826087, 0.3756051538341304, 1.1797436263554544], '85 / 280 / 1': [10083.158520302464, 0.2391304347826087, 0.3756051538341304, 1.1797436263554544], '85 / 290 / 1': [10083.158520302464, 0.2391304347826087, 0.3756051538341304, 1.1797436263554544], '90 / 100 / 1': [14700.485496993495, 0.34375, 1.7343450224343495, 1.8238344342720756], '90 / 110 / 1': [16031.586447069038, 0.36666666666666664, 2.1230897958906905, 2.086298111734322], '90 / 120 / 1': [16031.586447069038, 0.36666666666666664, 2.1230897958906905, 2.086298111734322], '90 / 130 / 1': [16031.586447069038, 0.36666666666666664, 2.1230897958906905, 2.086298111734322], '90 / 140 / 1': [15963.540894015501, 0.36666666666666664, 2.108966253462301, 2.075173885676265], '90 / 150 / 1': [15963.540894015501, 0.36666666666666664, 2.108966253462301, 2.075173885676265], '90 / 160 / 1': [15963.540894015501, 0.36666666666666664, 2.108966253462301, 2.075173885676265], '90 / 170 / 1': [16593.81615830524, 0.36666666666666664, 2.2075292736830443, 2.192081750164473], '90 / 180 / 1': [15037.538314956604, 0.3055555555555556, 1.5773067868631574, 1.898799486188779], '90 / 190 / 1': [13585.204362575154, 0.2619047619047619, 1.118535393482617, 1.651990913887927], '90 / 200 / 1': [13524.862854179579, 0.2619047619047619, 1.1060166404387992, 1.644693770422164], '90 / 210 / 1': [13524.862854179579, 0.2619047619047619, 1.1060166404387992, 1.644693770422164], '90 / 220 / 1': [13346.308121375154, 0.25, 1.0423334379498697, 1.6036954175183071], '90 / 230 / 1': [11212.14942828775, 0.22916666666666666, 0.6082619072177881, 1.31517527151617], '90 / 240 / 1': [11212.14942828775, 0.22916666666666666, 0.6082619072177881, 1.31517527151617], '90 / 250 / 1': [11338.945481315779, 0.22916666666666666, 0.636544387719265, 1.3298300417153697], '90 / 260 / 1': [10954.872807758984, 0.22, 0.546138990505273, 1.284793992533621], '90 / 270 / 1': [10954.872807758984, 0.22, 0.546138990505273, 1.284793992533621], '90 / 280 / 1': [10954.872807758984, 0.22, 0.546138990505273, 1.284793992533621], '90 / 290 / 1': [10954.872807758984, 0.22, 0.546138990505273, 1.284793992533621], '95 / 100 / 1': [18777.227806944637, 0.3181818181818182, 3.6799430276374294, 2.5230550010150288], '95 / 110 / 1': [18695.56838010265, 0.3181818181818182, 3.6580122680333607, 2.5078104112568105], '95 / 120 / 1': [18695.56838010265, 0.3181818181818182, 3.6580122680333607, 2.5078104112568105], '95 / 130 / 1': [18695.56838010265, 0.3181818181818182, 3.6580122680333607, 2.5078104112568105], '95 / 140 / 1': [20057.80818652715, 0.35, 4.343538645214146, 2.9111716269010235], '95 / 150 / 1': [21196.040411844377, 0.3888888888888889, 5.1332490081597655, 3.2459887717917715], '95 / 160 / 1': [21196.040411844377, 0.3888888888888889, 5.1332490081597655, 3.2459887717917715], '95 / 170 / 1': [20159.020152668498, 0.35, 4.370134872456379, 2.96051616707706], '95 / 180 / 1': [19165.986150367793, 0.3333333333333333, 3.39796781091715, 2.6270254104336397], '95 / 190 / 1': [18435.23740702753, 0.34615384615384615, 2.9972486474058155, 2.4404870497289393], '95 / 200 / 1': [18435.23740702753, 0.34615384615384615, 2.9972486474058155, 2.4404870497289393], '95 / 210 / 1': [18435.23740702753, 0.34615384615384615, 2.9972486474058155, 2.4404870497289393], '95 / 220 / 1': [17536.827507705115, 0.32142857142857145, 2.6016103355890805, 2.2724664451730576], '95 / 230 / 1': [15491.011652667372, 0.34375, 1.9086505436159031, 1.8596632940417614], '95 / 240 / 1': [14093.276028184491, 0.29411764705882354, 1.52833860939573, 1.6560671957372743], '95 / 250 / 1': [14093.276028184491, 0.29411764705882354, 1.52833860939573, 1.6560671957372743], '95 / 260 / 1': [13282.282208902228, 0.2777777777777778, 1.285438413081148, 1.5451117369927652], '95 / 270 / 1': [13480.416396217082, 0.2777777777777778, 1.3304186645800533, 1.5706447048654564], '95 / 280 / 1': [13480.416396217082, 0.2777777777777778, 1.3304186645800533, 1.5706447048654564], '95 / 290 / 1': [13596.211055062411, 0.2777777777777778, 1.358844713464933, 1.5844126571861534]}


# # In[5]:


# end_time = df_4h.index[-1]

# brute_force_search_results_2 = {}
# for vwma_period in range(5,100,5):
#     for vwma_fisher_period in range(100,300,10):

#         key_name =  str(vwma_period) + " / " + str(vwma_fisher_period) + " / " + str(1)
#         results = trade(df_4h, end_time- 720*day, end_time - 500*day, True, True, False, 0.99925, vwma_period, vwma_fisher_period, 0)
#         wanted = ["Wallet", "Win Probability", "Trades' Return Mean",  "Weighted Ratio of Winning Trades Return Mean to Losing Trades Return Mean"]
#         values = [results.get(k) for k in wanted]
#         brute_force_search_results_2[key_name] = values
#         if  vwma_fisher_period == 290:
#             print(brute_force_search_results_2)
            
# print(brute_force_search_results_2)


# # In[6]:


# print(brute_force_search_results_2)


# # In[24]:


# fig = plt.figure(figsize= (15,15))
# ax = fig.add_subplot(projection='3d')
# keys = list(brute_force_search_results.keys()) 
# x = []
# y= []
# z = []
# for every in keys:
#     z_ax = brute_force_search_results[every][3]     
#     z.append(z_ax)
#     key = every.split(" / ")
#     x_ax = int(key[0])
#     x.append(x_ax)
#     y_ax = int(key[1])
#     y.append(y_ax)
#     ax.set_xticks(range(0,100,10))
#     ax.set_yticks(range(0, 100,10))
#     ax.scatter(x_ax, y_ax, z_ax, marker='o', color = 'k')
# keys_2 = list(brute_force_search_results_2.keys()) 
# for every in keys_2:
#     z_ax = brute_force_search_results_2[every][3]     
#     z.append(z_ax)
#     key = every.split(" / ")
#     x_ax = int(key[0])
#     x.append(x_ax)
#     y_ax = int(key[1])
#     y.append(y_ax)
#     ax.set_xticks(range(0,100,10))
#     ax.set_yticks(range(0, 300,10))
#     ax.scatter(x_ax, y_ax, z_ax, marker='o', color = 'k') 
    
 
    
"""ax = fig.add_subplot(111, projection='3d')
keys = list(brute_force_search_results.keys())
x = []
y= []
z = []
for every in keys:
    z_ax = brute_force_search_results[every][-1]     
    z.append(z)
    key = every.split(" / ")
    x_ax = int(key[0])
    x.append(x_ax)
    y_ax = int(key[1])
    y.append(y_ax)
    ax.set_xticks(range(0,100,10))
    ax.set_yticks(range(0, 100,10))
    ax.scatter(x_ax, y_ax, z_ax, marker='o', color = 'k')"""


# In[43]:


#trade(df_1d, end_time- 1500*day, end_time , True, True, True, 0.9995,20 ,170, 0.03)


# In[ ]:




