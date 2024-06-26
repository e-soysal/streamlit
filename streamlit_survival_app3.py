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
    if f > 0:
        P = np.exp(-2*f *w_0)
    else:
        P=1
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

def mu_x(x, a,b, mu_bm):
    if np.isscalar(x):
        if x > a:
            return mu_bm
        else:
            return  -b*(x-a) ** 2 +mu_bm
    else:
        # Create an array to store the results
        result = np.ones_like(x)*mu_bm
        # Set values according to the condition
        result[x < a] = -b*(x[x<a]-a) ** 2 +mu_bm
        return result

def prob_sig_function(x, a, b, w_0, mu_0, sigma_0, years):
    mu = mu_x(x,a,b, mu_0)
    transition_growth = (mu_0-x)*years
    w_year = w_0+transition_growth
    f_0 = mu / sigma_0**2    
    return np.log(ruin_prob(f_0, w_year))

def main():
    # User inputs
    mitigation_year = 628
    w_0 = 17527.18851
    
    # Display results
    hist_data = { 'year': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                 'GDP': [9968.671891, 10192.94482, 10440.52934,	10553.45026, 10776.47927, 11138.26003, 11246.50192, 11398.57076, 11668.42089, 12105.37947, 12495.37837, 12969.54552, 13464.8027, 13658.73942, 13391.3059, 13899.49329, 14263.1084, 14535.86611, 14820.27166, 15139.44768, 15461.1147, 15777.59079, 
                         16186.46995, 16588.58729, 16877.47189, 16213.37544, 17091.35075, 17527.18851]}
    hist_data = pd.DataFrame(hist_data)
    
    growth = hist_data['GDP'].diff()
    mu_bm = np.mean(growth)
    sigma_bm = np.std(growth)
    f = mu_bm/sigma_bm**2
    
    st.write('Choose the shape of the drift mu(x) by adjusting shape and location:')
    a = st.slider('Impact point (left to right)', value=1100, min_value=0, max_value=800, step=1)
    b_input = st.slider('Shape (flat to steep):', value=2.0, min_value=-1.0, max_value=10.0, step=0.1)
    st.write('Choose a the yearly expenditure on mitigation in USD per capita per year:')
    mitigation = st.slider(':orange-background[Mitigation expenditure in USD per year:]', value=mitigation_year, min_value=0, max_value=1000, step=10, label_visibility = "collapsed" )    

    b = b_input/10000
    
    
    # Find transition GDP until 2032
    x = np.arange(0,11)+2022
    transition_growth = (mu_bm-mitigation)*np.ones(len(x))
    transition_growth[0] = 0
    transition_GDP = w_0 + transition_growth.cumsum()
    
    # Calculate new ruin probability 2032
    mu_mitigation = mu_x(mitigation, a, b, mu_bm)
    P = ruin_prob(mu_mitigation/sigma_bm**2, transition_GDP[-1])
    st.write('Probability of ruin: ', P)
    
    # Simulate path in future
    x_2 = np.arange(0,90)+x[-1]
    rd = np.random.normal(mu_mitigation, sigma_bm, len(x_2))
    rd[0] = 0
    random_path = transition_GDP[-1] +  rd.cumsum()
    
    # Plot wealth projections
    fig, ax = plt.subplots(2,1, figsize=(6, 10))
    ax[1].plot(hist_data.year, hist_data.GDP, label = "Historical", color = 'black')
    ax[1].plot(x, transition_GDP, label = "Transition path", color = 'red')
    ax[1].plot(x_2, random_path, label = "Simulated path", color = 'red')
    ax[1].scatter(x[-1], transition_GDP[-1], label = "", color = 'Black')
    ax[1].annotate('w_0', xy=(x[-1], transition_GDP[-1]), xytext=(x[-1]-5, transition_GDP[-1]-1000),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax[1].set_ylim(0,np.max([np.max(transition_GDP),np.max(hist_data.GDP[:]),np.max(random_path)])+1000)
    ax[1].set_xlabel('Year')
    ax[1].set_title('GDP per capita [PPP, 2017 USD]')
    ax[1].legend()
    ax[1].grid(True)

    
    x_input = np.linspace(0, 1200, 100)
    mu_input = mu_x(x_input, a, b, mu_bm)
    
    # Find optimal mitigation
    X_init = mitigation_year # Initial guess
    
    # Create a partial function with fixed arguments
    partial_function = partial(prob_sig_function, a=a, b=b, w_0=w_0, mu_0=mu_bm, sigma_0=sigma_bm, years=len(x)-1)
    result = minimize(partial_function, X_init)
    if result.x[0]>0:
        st.write('Optimal spending on mitigation: ', np.round(result.x[0],0), 'Optimal probability of ruin: ', np.exp(result.fun))
        ax[0].scatter(result.x[0], mu_x(result.x[0], a,b, mu_bm), label = "Optimal strategy", color = 'blue')
    else:
        st.write('Optimal spending on mitigation:        Optimal ruin probability: ')
        
    ax[0].scatter(mitigation, mu_x(mitigation,a,b, mu_bm), label = "Chosen strategy", color = 'Black')
    ax[0].fill_between([mitigation_year-50, mitigation_year+50], [mu_bm+50,mu_bm+50],[0,0] ,color = 'green', label = '2 degrees required spending', alpha=.3)
    ax[0].fill_between([125-5, 125+5], [mu_bm+50,mu_bm+50],[0,0] ,color = 'red', label = 'Current spending', alpha=.3)
    ax[0].plot([a, a], [mu_bm-25,mu_bm+25], color = 'black', linestyle = "--", label = '')
    ax[0].annotate('a', xy=(a, mu_bm-25))
    ax[0].plot(x_input, mu_input, color = 'black')
    
    ax[0].legend()
    ax[0].set_title('Drift, mu(X)')
    ax[0].set_xlabel('Mitigation in USD per year')
    ax[0].set_ylim(0,mu_bm+50)
     
    st.pyplot(plt)
    plt.show()

if __name__ == '__main__':
    main()
