"""
Created on 10 05 2024

@author: Emilie Rosenlund Soysal
"""
import random
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import math
from scipy.stats import linregress
from scipy.optimize import minimize
from functools import partial

def ruin_prob(mu,sigma, w_0):
    """Ruin probability in infinite time"""
    if mu > 0:
        P = np.exp(-2*(mu/sigma**2 )*w_0)
    else:
        P=1
    return P

def ruin_prob_T(mu, sigma, w_0, T):
    """Ruin probability within finite time horizon T"""
    P = 1- ( norm.cdf( (w_0+mu*T)/(sigma*math.sqrt(T)),loc=0,scale=1) 
            - ruin_prob(mu,sigma,w_0)*norm.cdf( (-w_0+mu*T)/(sigma*math.sqrt(T)),loc=0,scale=1))
    return P

def main():
    # Display results
    hist_data = { 'year': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                 'GDP': [9968.671891, 10192.94482, 10440.52934,	10553.45026, 10776.47927, 11138.26003, 11246.50192, 11398.57076, 11668.42089, 12105.37947, 12495.37837, 12969.54552, 13464.8027, 13658.73942, 13391.3059, 13899.49329, 14263.1084, 14535.86611, 14820.27166, 15139.44768, 15461.1147, 15777.59079, 
                         16186.46995, 16588.58729, 16877.47189, 16213.37544, 17091.35075, 17527.18851]}
    #Starting wealth
    w_0 = hist_data['GDP'][-1]
    
    #Store data in dataframe
    hist_data = pd.DataFrame(hist_data)

    #Calculate GDP growth, drift and volatility of growth
    growth = hist_data['GDP'].diff()
    mu_bm = np.mean(growth)
    sigma_bm = np.std(growth)

    # Input from users
    # st.write('Choose the damage by year 2100 in percent of GDP:')
    # damage_2100 = st.slider('Damage', value=16.5, min_value=0.0, max_value=100.0, step=0.5)
    
    #st.write('Choose the the drift under impact of climate change:')
    #mu_cc = st.slider('Drift', value=-20, min_value=-200, max_value=280, step=1)

    # st.write('Choose the shift year:')
    year_i = st.slider('Choose onset year', value=2023, min_value=2023, max_value=2099, step=1)
    
    #st.write('Choose the shift year damage level:')
    #damage_i = st.slider('Shift damage level', value=0.0, min_value=0.0, max_value=20.0, step=0.1)
    damage_i=0
    
    #st.write('Choose the time horizon:')
    #T = st.slider('Time horizon', value=500, min_value=100, max_value=5000, step=50)
    T=500
    
    # Define two periods - pre and post shift
    x_1 = np.arange(0,year_i-2022)+2022 
    x_2 = np.arange(0,T+2022-year_i)+year_i
    x = np.append(x_1, x_2)
    
    # Benchmark - expected GDP if there was no such thing as CC.
    g_bm = mu_bm*np.ones(len(x))
    g_bm[0]=0
    GDP_bm = w_0 + g_bm.cumsum()
    w_bm = GDP_bm[2100-2022]
    print('w_bm', GDP_bm[2100-2022], (2100-2022)*mu_bm+w_0)
    
    # With damages - expected GDP if under input assumptions.
    # Pre-shift
    mu_1 = mu_bm*(1-damage_i/100)
    #mu_1 = (w_1-w_0)/(2100 - 2022)
    g_1 = mu_1*np.ones(len(x_1))
    g_1[0]=0
    GDP_1 = w_0 + g_1.cumsum()
    w_1 = GDP_1[-1] 
    
    #Post shift
    damage_IPCC=16.5
    w_2100 = w_bm*(1-damage_IPCC/100)
    
    mu_2 = (w_2100 - w_1)/(2100-year_i)
    print('w_0, w_1, w_bm, w_2100', w_0, w_1, w_bm, w_2100)
    print('mu_bm, mu_1, mu_2', mu_bm, mu_1, mu_2)
    
    g_2 = mu_2*np.ones(len(x_2))
    g_2[0]=0
    GDP_2 = GDP_1[-1] + g_2.cumsum()
    print(len(x_1), len(x_2),len(GDP_1), len(GDP_2))
    
    #With uncertainty
    GDP_post = np.append(GDP_1, GDP_2)
    t = np.arange(len(x))
    t_sqrt= np.sqrt(t)
    std = sigma_bm*t_sqrt
    GDP_low = GDP_post-std
    GDP_high = GDP_post+std
    
    GDP_IPCC_low =w_bm*(1-23/100)
    GDP_IPCC_high = w_bm*(1-10/100)
    
    # Calculate new ruin probability 2032
    #P = ruin_prob_T(mu_cc,sigma_bm, w_1,T)
    #st.write('Probability of ruin: ', P)
    #   damage =  (GDP_2bm[2100-year_i]- GDP_2[2100-year_i] )/ GDP_2bm[2100-year_i]
    # st.write('GDP damage in percent by 2100: ', damage_2100)
    
    #print('GDP damage in percent by 2100: ', round(damage_2100,0))
        
    # Plot wealth projections
    fig, ax = plt.subplots(1,2, figsize=(10, 5), sharex = True)
    ax[0].plot(hist_data.year, hist_data.GDP, label = "Historical", color = 'black')
    ax[0].plot(np.append(x_1, x_2), GDP_bm, label = "Benchmark projection", color = 'black', linestyle = '--')
    ax[0].plot(x_1, GDP_1, label = "Pre-shift", color = 'purple')
    ax[0].plot(x_2, GDP_2, label = "Post-shift", color = 'blue')
    ax[0].fill_between(np.append(x_1,x_2), np.append(GDP_1,GDP_2), GDP_bm, alpha = 0.2, label= 'Expected damage' )
    # ax[0].fill_between(x, GDP_low,GDP_high, alpha = 0.2, label= 'Uncertainty bands')
    
    #ax[0].plot([year_i, year_i], [GDP_2[0], GDP_bm[2022+T-year_i]], color = 'black', linestyle = ':')
    
    ax[0].plot([2100, 2100], [GDP_IPCC_high, GDP_IPCC_low],
               color = 'black', linestyle = '-', linewidth= 2.5)
    
    ax[0].scatter([year_i], GDP_1[-1], label = "", color = 'Black')
    
#    ax[0].annotate('Expected damage in 2100: ' + str(round(damage*100,0))[0:2] + '%', xy=(2100, (GDP_2[2100-year_i]+GDP_2bm[2100-year_i])/2),
#                   xytext=(2100+10, (GDP_2[2100-year_i]+GDP_2bm[2100-year_i])/2),
#                   arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    ax[0].annotate('IPCC 4C forecast', xy=(2100, w_bm*(1-16.5/100)),
                   xytext=(2100+10, (GDP_2[2100-year_i]+GDP_bm[year_i-2022])/2+7000),
                   arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax[0].set_ylim(0,82000)
    ax[0].set_xlim(1995,2200)
    ax[0].set_xlabel('Year')
    ax[0].set_title('GDP per capita [PPP, 2017 USD]')
    ax[0].legend()
    ax[0].grid(True)

    # Discounted utility
    x = np.append(x_1, x_2)
    GDP_damages = GDP_bm - np.append(GDP_1,GDP_2)
    beta = np.cumprod(np.ones(len(x))*0.96)
    GDP_discounted = np.multiply(GDP_damages,beta)
    E_NPV =np.round(np.sum(GDP_discounted))
    ax[1].plot(x, beta )
    ax[1].bar(x[0:200], GDP_discounted[0:200] )
    ax[1].set_ylim(0,820)
    ax[1].set_xlim(1995,2200)
    ax[1].set_title('Expected discounted damage')
    ax[1].annotate('NPV: ' + E_NPV, xy=(2100, 700), xytext=(2160, 700))
    st.pyplot(plt)
    plt.show()

if __name__ == '__main__':
    main()
