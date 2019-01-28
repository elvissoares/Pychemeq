import numpy as np
from scipy import optimize
# Define the log10 of the equilibrium constants
logK = {'H': -1.464, 'a1': -6.363, 'a2': -10.329, 'w': -13.997, 'sp': -8.48, \
'CaH': 1.26, 'CaC': 3.15, 'CaOH': 1.3, 'NaOH': -14.18, 'NaCO3-': 1.27, 'NaHCO3':-0.25 }
# Partial pressure of CO2 in air
logPCO2 = -3.455
# Define the concentration of each element
c = {'H+': 0., 'OH-': 0., 'CO2': 0., 'CO3--': 0., 'HCO3-': 0., 'Ca++': 0., \
'CaOH+': 0., 'CaHCO3+': 0., 'CaCO3(aq)': 0., 'Na+': 0., 'NaOH': 0.,'NaCO3-': 0., 'Cl-': 0 }
# Calculate the ionic strength
def ionic_strength(c):
    return 0.5*(4*np.power(10,c['Ca++'])+np.power(10,c['CaHCO3+'])\
            +np.power(10,c['CaOH+'])+np.power(10,c['H+'])\
            +4*np.power(10,c['CO3--'])+np.power(10,c['HCO3-'])\
            +np.power(10,c['OH-']) + np.power(10,c['Na+']) \
            + np.power(10,c['Cl-']) + np.power(10,c['NaCO3-']) )
# Calculate the activity coefficients
loggamma = {'0': 0., '+': 0., '-': 0., '++': 0., '--': 0.}
def calculate_loggamma(m):
    b = 0.1; A = 0.5
    loggamma['0'] = b*m
    loggamma['+'] = loggamma['-'] = -A*(1)*(np.sqrt(m)/(1+np.sqrt(m))-0.2*m)
    loggamma['++'] = loggamma['--'] = 4*loggamma['+']
    return
# The aqueous NaHCO3 equilibrium reactions for an open system
def NaHCO3_equilibrium(x,cNaHCO3):
    c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]; \
    c['HCO3-'] = x[4]; c['Na+'] = x[5]; c['NaOH'] = x[6]; c['NaCO3-'] = x[7];\
    c['NaHCO3'] = x[8]; 
    cNaT = cNaHCO3; #This came from sodium carbonate
    # ionic strength
    I = 0.5*(np.power(10,c['H+'])\
            +4*np.power(10,c['CO3--'])+np.power(10,c['HCO3-'])\
            +np.power(10,c['OH-']) + np.power(10,c['Na+']) + np.power(10,c['NaCO3-']) )
    calculate_loggamma(I)
    # carbonate-CO2 equilibrium
    Reaction = [None]*9
    Reaction[0] = c['CO2'] - logPCO2 - (logK['H']-loggamma['0'])
    Reaction[1] = c['HCO3-'] + c['H+'] - (logK['a1']+loggamma['0'] -loggamma['+']-loggamma['-'])- c['CO2']
    Reaction[2] = c['CO3--'] + c['H+'] - (logK['a2']-loggamma['--']) - c['HCO3-']
    Reaction[3] = c['OH-'] + c['H+'] - (logK['w'] + loggamma['0'] - loggamma['+']-loggamma['-'])
    # Sodium reactions
    Reaction[4] =  c['Na+'] + loggamma['+'] + logK['NaOH'] + c['OH-']+ loggamma['-'] - c['NaOH'] - loggamma['0']
    Reaction[5] =  c['Na+'] + c['CO3--'] + (logK['NaCO3-']+loggamma['--']) - c['NaCO3-']
    Reaction[6] =  c['Na+'] + c['HCO3-'] + (logK['NaHCO3']-loggamma['0'] +loggamma['+']+loggamma['-']) - c['NaHCO3']
    Reaction[7] = cNaT - np.power(10,c['Na+']) - np.power(10,c['NaCO3-'])- \
    np.power(10,c['NaOH'])- np.power(10,c['NaHCO3'])
    # Charge Conservation
    Reaction[8] = np.power(10,c['H+']) + np.power(10,c['Na+']) - np.power(10,c['HCO3-'])\
    - 2*np.power(10,c['CO3--']) - np.power(10,c['OH-']) - np.power(10,c['NaCO3-'])
    return Reaction
# Define the system of chemical equilibrium equations to solve
def Mixture(x,cCaT,DIC,cNaT,cClT):
    c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]; \
    c['HCO3-'] = x[4]; c['Ca++'] = x[5]; c['CaOH+'] = x[6]; c['CaHCO3+'] = x[7];\
    c['CaCO3(aq)'] = x[8]; c['Na+'] = x[9]; c['NaOH'] = x[10]; c['NaCO3-'] = x[11]; \
    c['NaHCO3'] = x[12]; c['Cl-'] = np.log10(cClT);

    m = ionic_strength(c)
    calculate_loggamma(m)
    # Here we go, to define the reactions to carbonate-CO2 equilibrium
    Reaction = [None]*13
    Reaction[0] = c['HCO3-'] + c['H+'] - (logK['a1']+loggamma['0'] -loggamma['+']-loggamma['-'])- c['CO2']
    Reaction[1] = c['CO3--'] + c['H+'] - (logK['a2']-loggamma['--']) - c['HCO3-']
    Reaction[2] = c['OH-'] + c['H+'] - (logK['w'] + loggamma['0'] - loggamma['+']-loggamma['-'])
    # Calcium reactions
    Reaction[3] = c['CaHCO3+'] - (logK['CaH']+loggamma['++']) - c['Ca++'] - c['HCO3-']
    Reaction[4] = c['CaCO3(aq)'] - (logK['CaC']-loggamma['0']+loggamma['++']+loggamma['--']) - c['Ca++'] - c['CO3--']
    Reaction[5] = c['CaOH+'] - (logK['CaOH']+loggamma['++']) - c['Ca++'] - c['OH-']
    # Charge conservation
    Reaction[6] = np.power(10,c['H+']) + 2*np.power(10,c['Ca++']) \
    + np.power(10,c['CaHCO3+']) + np.power(10,c['CaOH+']) - np.power(10,c['HCO3-'])\
     - 2*np.power(10,c['CO3--']) - np.power(10,c['OH-']) + np.power(10,c['Na+']) \
     - np.power(10,c['Cl-']) - np.power(10,c['NaCO3-'])
    # Total calcium concentration
    Reaction[7] = cCaT - np.power(10,c['Ca++']) - np.power(10,c['CaHCO3+']) \
    - np.power(10,c['CaCO3(aq)']) - np.power(10,c['CaOH+'])
    # Total carbon concentration
    Reaction[8] =  DIC - np.power(10,c['CO2']) - np.power(10,c['CO3--']) \
    - np.power(10,c['HCO3-'])- np.power(10,c['CaHCO3+'])- np.power(10,c['CaCO3(aq)'])\
    - np.power(10,c['NaCO3-'])- np.power(10,c['NaHCO3'])
    # Sodium reactions
    Reaction[9] =  c['Na+'] + logK['NaOH'] - c['NaOH'] - c['H+']
    Reaction[10] =  c['Na+'] + c['CO3--'] + (logK['NaCO3-']+loggamma['--']) - c['NaCO3-']
    Reaction[11] =  c['Na+'] + c['HCO3-'] + (logK['NaHCO3']-loggamma['0'] +loggamma['+']+loggamma['-']) - c['NaHCO3']
    Reaction[12] = cNaT - np.power(10,c['Na+']) - np.power(10,c['NaCO3-'])- \
        np.power(10,c['NaOH'])- np.power(10,c['NaHCO3'])
    return Reaction

def calculate_IS(pCa,pCO3):
    # NaHCO3 equilibrium  ##########
    cNaT = np.power(10.0,-pCO3)/2.0
    solNaHCO3 = optimize.root(NaHCO3_equilibrium, [-1.0] * 9, args=cNaT, method='hybr')
    DIC = np.power(10,c['CO2'])+np.power(10,c['CO3--'])+ np.power(10,c['HCO3-']) + np.power(10,c['NaCO3-'])
    cCaT = np.power(10.0,-pCa)/2.0
    cClT = 2*cCaT
    sol = optimize.root(Mixture, [-0.1] * 13, args=(cCaT,DIC,cNaT,cClT), method='hybr')
    S = np.power(10,c['Ca++']+loggamma['++'])*np.power(10,c['CO3--']+loggamma['--'])/np.power(10.0,logK['sp'])
    return np.log10(S)

pCaarray = np.arange(0.001,3.0,0.1)
pCO3array = np.arange(0.001,3.0,0.1)

# IS = [[calculate_IS(pCa,pCO3) for pCa in pCaarray] for pCO3 in pCO3array]

cNaHCO3 = 0.5 #mol/L
cCaCl2 = 0.5 #mol/L

cNaHCO3adao = 0.15 #mol/L
cCaCl2adao = 0.05 #mol/L

mCaCl2 = 147.02 #110.98 normal if dihydrated -> 147.02 g/mol 
mNaHCO3 = 84.006 #g/mol
# Concentrações iniciais (g/L)
cNaHCO3jose = 1.2275/mNaHCO3 #mol/L
cCaCl2jose =0.7375/mCaCl2 #mol/L

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# fig, ax = plt.subplots()
# CS = ax.contour(-pCaarray, -pCO3array, IS, [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5], cmap=plt.cm.winter)
# ax.clabel(CS, inline=1, fontsize=10, fmt='%1.1f')
# ax.plot(-pCaarray, -pCO3array, '--', color='silver')
# ax.plot(np.log10(cCaCl2), np.log10(cNaHCO3), 'o', label='calculated')
# ax.plot(np.log10(cCaCl2adao), np.log10(cNaHCO3adao), 'o', label='calculated')
# ax.plot(np.log10(cCaCl2jose), np.log10(cNaHCO3jose), 'o', label='calculated')
# ax.set_title('Saturation Index, SI')
# ax.set_xlabel('log(cCaT)')
# ax.set_ylabel('log(cCO3T)')
# ax.set_aspect('equal')
# plt.show()


# Calcite Growth calculation
# k2 = 4.6e-2 #nm/s
# S = np.sqrt(np.power(10.0,IS))
# rdot = k2*(S-1)**2

# fig, ax = plt.subplots()
# CS = ax.contour(-pCaarray, -pCO3array, rdot, [0.1,1,10,100,1000], colors='k')
# ax.clabel(CS, inline=1, fontsize=10, fmt='%1.1f nm/s')
# ax.plot(-pCaarray, -pCO3array, '--', color='silver')
# ax.plot(np.log10(cCaCl2), np.log10(cNaHCO3), 'ro', label='calculated')
# ax.plot(np.log10(cCaCl2adao), np.log10(cNaHCO3adao), 'o', label='calculated')
# ax.plot(np.log10(cCaCl2jose), np.log10(cNaHCO3jose), 'o', label='calculated')
# ax.set_xlabel('log(cCaT)')
# ax.set_ylabel('log(cCO3T)')
# ax.set_aspect('equal')
# plt.show()

## Find IS 


IS = 0.75

def pCO3fromIS(x,pCa):
    return (IS - calculate_IS(pCa,x))

pCO3ISarray = [optimize.brentq(pCO3fromIS, 0, 3.0, args=pCa) for pCa in pCaarray]
pCO3ISarray = np.array(pCO3ISarray)
# plt.plot(-pCaarray, pCO3ISarray, label='SI=0.7')
# plt.legend()
# plt.xlim(-3.0,0.0)
# plt.ylim(-3.0,0.0)
# plt.xlabel("log(cCaT)")
# plt.ylabel("log(cCO3T)")
# plt.show()

# acute constants
kCa = 6.4e6
kCO3 = 4.7e6
kkn = 0.0

a = 0.31 # nm

def Rkn(pCa,pCO3):
    Ca = np.power(10,-pCa)
    CO3 = np.power(10,-pCO3)
    return (kCa*Ca*kCO3*CO3/(kCa*Ca + kCO3*CO3) - kkn)

rdot = a*Rkn(pCaarray,pCO3ISarray)
ratio = -(pCaarray-pCO3ISarray)

plt.plot(ratio, rdot, label='SI=0.75')
plt.legend()
#plt.xlim(-3.0,0.0)
#plt.ylim(-3.0,0.0)
plt.xlabel("log(cCaT/cCO3)")
plt.ylabel("$\dot{r}$ nm/s")
plt.show()
