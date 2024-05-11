"""
Created on 10 05 2024

@author: Emilie Rosenlund Soysal
"""
import random
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def ruin_prob(f, w_0):
    """Ruin probability"""
    P = np.exp(-2*f *w_0)
    return P
    
def sigmoid(x,a,b):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-(x-a)*b))

def main():
    st.write("Benchmark process: $\mu$: 275, $\sigma$: 265, $w_0$: 17500")
    st.write("To stay below 2 degrees, we need to invest 5 percent of GDP")
# User inputs
    w_0 = 17500
    mu = st.slider('Drift, $\mu$:', value=275, min_value=0, max_value=500, step=5)
    sigma = st.slider('Volatility, $\sigma$:', value=265, min_value=0, max_value=3000, step=5)
    x = np.linspace(0, 1000, 100) 
    f = mu/sigma^2
    P = ruin_prob(mu/sigma^2,w_0)
    st.write('Ruin probability', P)
    
    # Display results
    
    fig, ax = plt.subplots(1,2, figsize=(10, 4))
    ax[0].plot(x, g, label = "Impact factor", color = 'black')
    ax[0].set_xlabel('Mitigation expenditure, X')
    ax[0].set_ylabel('g(X)')
    ax[0].legend()
    ax[0].set_title('Fig. 2A: Odds function')
    ax[0].grid(True)

    # Plotting the ruin probability
    st.pyplot(plt)
     
if __name__ == '__main__':
    main()
