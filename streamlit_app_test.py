"""
Created on 10 05 2024

@author: Emilie Rosenlund Soysal
"""
import random
import streamlit as st
import numpy as np

st.title("This is survival prob simulation 1")

def ruin_prob(f, w_0):
    """Ruin probability"""
    P = np.exp(-2*f *w_0)
    return P
    
def sigmoid(x,a,b):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x*a+b))

def main():
    # User inputs
    mu = st.slider('Drift:', value=0.95, min_value=0.00, max_value=1.00, step=0.01)
    sigma = st.slider('Volatility:', value=0.05, min_value=0.00, max_value=1.00, step=0.01)
    
    if st.button('Simulate'):
        prob = ruin_prob(mu/sigma**2, 100)
        # Display results
        st.write(prob)
      
if __name__ == '__main__':
    main()
