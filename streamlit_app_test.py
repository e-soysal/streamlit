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
    st.write("Benchmark process: $\mu$: 275, $\sigma$: 265, $w_0$: 17500")
    # User inputs
    f = 275/265**2
    w_0 = 17500
    a = st.slider('a:', value=0.95, min_value=0.00, max_value=5.00, step=0.01)
    b = st.slider('b:', value=0.05, min_value=0.00, max_value=5.00, step=0.01)
    x = np.linspace(0, 6000, 100)  # Generate 100 points between -10 and 10
    g = sigmoid(x,a,b)
    P = ruin_prob(g*f,w_0)
    # Display results
    
    fig, ax = plt.subplots(1,2, figsize=(10, 4))
    ax[0].plot(x, f*g, label = "Impact factor", color = 'black')
    ax[0].set_xlabel('Mitigation expenditure, X')
    ax[0].set_ylabel('g(X)*f')
    ax[0].legend()
    ax[0].set_title('Fig. 2A: Odds function')
    ax[0].grid(True)

    # Plotting the ruin probability
    # and the minimum points
    min_index_1 = np.argmin(P)
    min_x_1 = x[min_index_1]
    min_y_1 = P[min_index_1]
    ax[0].set_xticks([])  # Remove x-axis numbers
    ax[0].set_yticks([])  # Remove y-axis numbers
    st.pyplot(plt)
     
if __name__ == '__main__':
    main()
