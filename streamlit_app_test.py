"""
Created on 10 05 2024

@author: Emilie Rosenlund Soysal
"""
import random
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

def ruin_prob(f, w_0):
    """Ruin probability"""
    P = np.exp(-2*f *w_0)
    return P
    
def sigmoid(x,a,b):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-(x-a)*b))

def find_ruin(vector):
    # Find the index of the first element that is 0 or smaller and set it and all following elements to 0.
    if np.any(vector <= 0):
        first_negative_index = np.argmax(vector <= 0)
        vector[first_negative_index:] = 0
    return vector

def main():
    w_0 = 17527.18851
    hist_data = { 'year': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                 'GDP': [9968.671891, 10192.94482, 10440.52934,	10553.45026, 10776.47927, 11138.26003, 11246.50192, 11398.57076, 11668.42089, 12105.37947, 12495.37837, 12969.54552, 13464.8027, 13658.73942, 13391.3059, 13899.49329, 14263.1084, 14535.86611, 14820.27166, 15139.44768, 15461.1147, 15777.59079, 
                         16186.46995, 16588.58729, 16877.47189, 16213.37544, 17091.35075, 17527.18851]}
    hist_data = pd.DataFrame(hist_data)
    
    growth = hist_data['GDP'].diff()
    mu_bm = np.mean(growth)
    sigma_bm = np.std(growth)
    
    st.write("Initial output $w_0$:", int(np.round(w_0)), " USD")
    st.write("Benchmark drift $\mu_{BM}$:", int(np.round(mu_bm)), " USD per year")
    st.write("Benchmark volatility $\sigma_{BM}$:", int(np.round(sigma_bm)), " USD per year^(1/2)")
    # User inputs
    mu = st.slider('Drift, $\mu$:', value=int(np.round(mu_bm,0)), min_value=-300, max_value=500, step=1)
    sigma = st.slider('Volatility, $\sigma$:', value=int(np.round(sigma_bm,0)), min_value=1, max_value=3000, step=1)
    
    f = mu/sigma**2
    if mu <= 0:
        P = 1
    else:    
        P = ruin_prob(f,w_0)
    st.write('Ruin probability: ', P)
    
    # Create a random path based on selected drift and volatility
    x = np.arange(0,100)+2022
    rd = np.random.normal(mu, sigma, len(x))
    rd[0] = 0
    random_path = w_0 +  rd.cumsum()
    random_path = find_ruin(random_path)
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(hist_data['year'], hist_data['GDP'])
    trend_GDP = slope * x + intercept
    
    # Plot
    fig, ax = plt.subplots(1,1, figsize=(6, 5))
    ax.plot(hist_data.year, hist_data.GDP, label = "Historical", color = 'black')
    ax.plot(x, trend_GDP, label = "Benchmark projection", color = 'black', linestyle = '--')
    ax.plot(x, random_path, label = "Simulation", color = 'red')
    ax.set_xlabel('Year')
    ax.legend()
    ax.set_ylim(0, np.max([random_path,trend_GDP]))
    ax.set_title('GDP per capita [PPP, 2017 USD]')
    ax.grid(True)

    # Plotting the ruin probability
    st.pyplot(plt)
     
if __name__ == '__main__':
    main()
