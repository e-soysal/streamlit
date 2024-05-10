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
    return 1 / (1 + np.exp(-x*a+b))

def main():
    st.title("Benchmark process: $\mu$: 275, $\sigma$: 265, $w_0$: 17500")
    # User inputs
    f = 275/265**2
    w_0 = 17500
    a = st.slider('Drift:', value=0.95, min_value=0.00, max_value=1.00, step=0.01)
    b = st.slider('Volatility:', value=0.05, min_value=0.00, max_value=1.00, step=0.01)
    x = np.linspace(0, 6000, 100)  # Generate 100 points between -10 and 10
    g = sigmoid(x,a,b)
    P = ruin_prob(g*f,w_0)
    if st.button('Simulate'):
        # Display results
        st.write(P)
      
if __name__ == '__main__':
    main()
