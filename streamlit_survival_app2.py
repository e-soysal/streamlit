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
from scipy.optimize import minimize
from functools import partial

def ruin_prob(f, w_0):
    """Ruin probability"""
    P = np.exp(-2*f *w_0)
    return P
    
def sigmoid(x,a,b):
    """Sigmoid function."""
    # a shifts left to right (neg <- and pos -> )
    # b flattens (Small b = flat, large b = steep)
    # x between -5 and 5.
    return 1 / (1 + np.exp(-(x-a)*b))

def find_ruin(vector):
    # Find the index of the first element that is 0 or smaller
    if np.any(vector <= 0):
        first_negative_index = np.argmax(vector <= 0)
        vector[first_negative_index:] = 0
    return vector

def prob_sig_function(x, scaling, a, b, w_0, mu_0, sigma_0, years):
    sig = sigmoid(x/scaling,a,b)
    transition_growth = (mu_0-x)*years
    w_year = w_0+transition_growth
    f_0 = mu_0 / sigma_0**2    
    return np.log(ruin_prob(sig*f_0, w_year))


def main():
    # User inputs
    mitigation_year = 628
    
    # Display results
    hist_data = { 'year': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                 'GDP': [9968.671891, 10192.94482, 10440.52934,	10553.45026, 10776.47927, 11138.26003, 11246.50192, 11398.57076, 11668.42089, 12105.37947, 12495.37837, 12969.54552, 13464.8027, 13658.73942, 13391.3059, 13899.49329, 14263.1084, 14535.86611, 14820.27166, 15139.44768, 15461.1147, 15777.59079, 
                         16186.46995, 16588.58729, 16877.47189, 16213.37544, 17091.35075, 17527.18851]}
    hist_data = pd.DataFrame(hist_data)
    w_0 = 17527.18851
    
    growth = hist_data['GDP'].diff()
    mu_bm = np.mean(growth)
    sigma_bm = np.std(growth)
    f = mu_bm/sigma_bm**2
    
    a = st.slider('Location (left -> right)', value=3.0, min_value=0.0, max_value=10.0, step=0.1)
    b = st.slider('Shape (flat -> steep):', value=1.00, min_value=0.01, max_value=2.0, step=0.01)
    mitigation = st.slider('Mitigation expenditure, X*, in USD per year:', value=mitigation_year, min_value=0, max_value=1000, step=10)    
    
    # Find transition GDP until 2032
    x = np.arange(0,11)+2022
    transition_growth = (mu_bm-mitigation)*np.ones(len(x))
    transition_growth[0] = 0
    transition_GDP = w_0 + transition_growth.cumsum()
    
    fig, ax = plt.subplots(1,2, figsize=(10, 4))
    ax[0].plot(hist_data.year, hist_data.GDP, label = "Historical", color = 'black')
    ax[0].plot(x, transition_GDP, label = "Transition path", color = 'red')
    ax[0].scatter(x[-1], transition_GDP[-1], label = "", color = 'Black')
    ax[0].annotate('w_0', xy=(x[-1], transition_GDP[-1]), xytext=(x[-1]-5, transition_GDP[-1]-1000),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax[0].set_ylim(0,np.max([np.max(transition_GDP),np.max(hist_data.GDP[:])])+1000)
    ax[0].set_xlabel('Year')
    ax[0].set_title('GDP per capita [PPP, 2017 USD]')
    ax[0].legend()
    ax[0].grid(True)

    scaling = 100
    sig_x = np.linspace(0, 1000, 100)
    sig = sigmoid(sig_x/scaling, a, b)
    g_mitigation = sigmoid(mitigation/scaling, a,b)
    # Calculate new ruin probability 2032
    P = ruin_prob(g_mitigation*f, transition_GDP[-1])
    st.write('Probability of ruin (after 2032): ', P)

    # Find optimal mitigation
    X_init = mitigation # Initial guess
    # Create a partial function with fixed arguments
    partial_function = partial(prob_sig_function, scaling=scaling, a=a, b=b, w_0=w_0, mu_0=mu_bm, sigma_0=sigma_bm, years=len(x))
    result = minimize(partial_function, X_init)
    st.write('Optimal spending on mitigation: ', result.x[0], 'Minimal ruin probability: ', np.exp(result.fun))
    
    ax[1].fill_between([mitigation_year-100, mitigation_year+100], [f,f],[0,0] ,color = 'green', label = '2 degrees required investments', alpha=.3)
    ax[1].plot([125, 125], [0,f], color = 'red', linestyle = "--", label = 'Current level')
    ax[1].plot(sig_x, sig*f, color = 'black')
    
    ax[1].scatter(mitigation, g_mitigation*f, label = "", color = 'Black')
    ax[1].annotate('f(X*)', xy=(mitigation,  g_mitigation*f), xytext=(mitigation-100, (g_mitigation-0.1)*f),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax[1].legend()
    ax[1].set_title('f(X)')
    ax[1].set_xlabel('Mitigation in USD per year')
     
    # Plotting the ruin probability
    st.pyplot(plt)
    plt.show()

if __name__ == '__main__':
    main()
