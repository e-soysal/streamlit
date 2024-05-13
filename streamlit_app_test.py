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
    hist_data = {'year': [ 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                 'cap_bUSD': [ 9167.878886, 9742.222509, 10289.9906, 10650.98046, 10517.4351, 10350.57569, 10206.75469, 10550.08958, 10950.57465, 11602.50899, 12111.50522, 12425.42771, 12874.83003, 13581.66477, 13636.91582, 13582.91239, 14033.21538, 14819.0123, 15670.84046, 16516.3381, 17319.8199, 17399.69746, 15764.18076, 16314.13008, 17058.61471, 17457.52613, 17841.61963, 18398.73892, 18923.97139, 19351.334, 20114.64002, 20851.69856, 21367.00667, 20467.34921, 21683.67718, 22344.93304]}
    hist_data = pd.DataFrame(hist_data)
    
    w_0 = 22344.93304
    
    growth = hist_data['cap_bUSD'].diff()
    mu_bm = np.mean(growth)
    sigma_bm = np.std(growth)
    
    st.write("Initial output $w_0$, billion USD:", int(np.round(w_0)))
    st.write("Benchmark drift $\mu_{BM}$, billion USD per year:", int(np.round(mu_bm)))
    st.write("Benchmark volatility $\sigma_{BM}$, billion USD per year^(1/2):", int(np.round(sigma_bm)))
    # User inputs
    mu = st.slider('Drift, $\mu$:', value=int(np.round(mu_bm,0)), min_value=-300, max_value=600, step=1)
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
    slope, intercept, r_value, p_value, std_err = linregress(hist_data['year'], hist_data['cap_bUSD'])
    trend_cap_bUSD = slope * x + intercept
    
    # Plot
    fig, ax = plt.subplots(1,1, figsize=(6, 5))
    ax.plot(hist_data.year, hist_data.cap_bUSD, label = "Historical", color = 'black')
    ax.plot(x, trend_cap_bUSD, label = "Benchmark projection", color = 'black', linestyle = '--')
    ax.plot(x, random_path, label = "Simulation", color = 'red')
    ax.set_xlabel('Year')
    ax.set_ylabel('billion USD (constant 2015)')
    ax.legend()
    ax.set_ylim(0, np.max([random_path,trend_cap_bUSD]))
    ax.set_title('Gross fixed capital formation')
    ax.grid(True)

    # Plotting the ruin probability
    st.pyplot(plt)
     
if __name__ == '__main__':
    main()
