# src/generate.py
#use equation 1, 2 and 3 to generate predicted summary stats 

import numpy as np

def forward_equations(alpha, nu, tau):

    if not (0.5 <= alpha <= 2) or not (0.5 <= nu <= 2) or not (0.1 <= tau <= 0.5):
        raise ValueError("Input values out of range")

    y = np.exp(-alpha * nu)
    #Eqn 1
    R_pred = 1 / (y + 1)
    #Eqn 2
    M_pred = tau + (alpha / (2 * nu)) * ((1 - y) / (1 + y))
    #Eqn 3
    ##V_pred = (alpha / (2 * nu**3)) * ((1 - 2 * alpha * nu * y - y**2) / (y + 1)**2)
    V_pred = (alpha / (2 * nu**3)) * ((1 + y - 2 * y * alpha * nu) / (1 + y)**2)

    #print(f"Debug (Generate Output): R_pred={R_pred}, M_pred={M_pred}, V_pred={V_pred}")

    return R_pred, M_pred, V_pred

   #Co-written w/ help of chatGPT
   
   
   


