"""
Dynamic PEM Fuel Cell (PEMFC) Model
Author: Iraj Moradpoor
Description:
Physics-based PEMFC performance model under variable load conditions.
"""
import numpy as np
import pandas as pd

R  = 8.314462618     # J/(mol·K)
F  = 96485.3329      # C/mol
z  = 2               # electrons for H2/O2 reaction
i_rated = 2
# Gas partial pressures (bar or atm; use the same units consistently)
P_amb = 1 # bar
P_a = 3.76
P_c = 12.5
lambda_air = 2
T = 353      # K, operating temperature
# Membrane (defaults typical for Nafion)
lambda_w = 14.0      # water content
L = 51e-4     # cm 
C = 3e-5            #V
d = 0.125            #A/cm2
# Geometry
A_cell = 250     # cm² active area (DEFAULT)
zeta_A = 1.1
zeta_C = 2.5
x_A = 0.0
x_C = 3.76
N_cell = 50
def P_H2O_sat():
    P_H2O = 10**-2.18 + (2.95e-2)*(T-273) - (9.18e-5)*(T-273)**2 + (1.44e-7)*(T-273)**3
    return P_H2O

def P_H2():
    x_H2O_A = P_H2O_sat()/P_a
    x_H2 = (1 - x_H2O_A) / (1 + (x_A/2) * ((1 + zeta_A) / (zeta_A - 1)))
    P_H2= x_H2*P_a
    return P_H2

def P_O2():
    x_H2O_C = P_H2O_sat()/P_c
    x_O2 = (1 - x_H2O_C) / (1 + (x_C/2) * ((1 + zeta_C) / (zeta_C - 1))) 
    P_O2 = x_O2*P_c
    return P_O2

def nernst_E():
    q = P_H2() * np.sqrt(P_O2())
    return 1.29 - 0.85e-3*(T - 298.15) + 4.31e-5*T* np.log(q)

def c_O2():
    C_O2 = P_O2() / (5.08e6 * np.exp(-498.0/T))
    return C_O2

def V_act(i):
    V_act = 0.9514 - 0.00312*T + 0.000187*T*np.log(i*A_cell) - 7.4e-5*T*np.log(c_O2())
    return V_act

def r_mem():
    return (181.6 * (1 + 0.03 * i + 0.062 * (T / 303)**2 * i**2.5)) / ((lambda_w - 0.634 - 3 * i) * np.exp(4.18 * (T - 303) / T))

def V_ohm(i):
    R_in = r_mem()*L/A_cell           # Ω·cm² (since t in cm, sigma in S/cm)
    return i * R_in*A_cell

def V_conc(i):
    V_con = -R*T/z/F*np.log(1-i/i_rated)
    return V_con

def V_cell(i):
    return nernst_E() - V_act(i) - V_ohm(i) - V_conc(i)

def power_W(i):
    return N_cell*V_cell(i)*i*A_cell

Current_density_array = np.linspace(0.01, 1.4, 50)
Voltage_array=[]
Power_array=[]

for i in Current_density_array:
    Voltage_array.append(V_cell(i))
    Power_array.append(power_W(i))

df = pd.DataFrame({
    "i_cell (A/cm2)": Current_density_array,
    "V_cell (V)": Voltage_array,
    "Power (W)": Power_array
    })
