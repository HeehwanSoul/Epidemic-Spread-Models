# -*- coding: utf-8 -*-
"""
Epidemic Spread Modeling with Diffential Equation

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd


""" 6 Models with the common coefficients """

# SI Model
def SI(y, t, N, beta):
    S, I = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N
    dRdt = 0
    return dSdt, dIdt

N = 51740000
I0 = 1
S0 = N - I0
y0 = S0, I0
beta = 0.2161

t = np.linspace(0, 240, 240)

ret = odeint(SI, y0, t, args=(N, beta))
S, I = ret.T

fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("SI Model")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# SIS Model
def SIS(y, t, N, beta, gamma):
    S, I = y
    dSdt = -beta * S * I / N + gamma * I 
    dIdt = beta * S * I / N - gamma * I 

    return dSdt, dIdt

N = 51740000
I0 = 1
S0 = N - I0
y0 = S0, I0
beta = 0.2161
gamma = 0.05

t = np.linspace(0, 240, 240)

ret = odeint(SIS, y0, t, args=(N, beta, gamma))
S, I = ret.T

fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("SIS Model")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# SIR Model
def SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I 
    return dSdt, dIdt, dRdt

N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
y0 = S0, I0, R0

beta = 0.2161
gamma = 0.05

t = np.linspace(0, 240, 240)

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R= ret.T
print(S[30])
fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("SIR Model")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# SIRS Model
def SIRS(y, t, N, beta, gamma, xi):
    S, I, R = y
    dSdt = -beta * S * I / N +xi*R
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I - xi*R
    return dSdt, dIdt, dRdt

N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
y0 = S0, I0, R0

beta = 0.2161
gamma = 0.05
xi = 0.2611 * 0.2161


t = np.linspace(0, 240, 240)

ret = odeint(SIRS, y0, t, args=(N, beta, gamma, xi))
S, I, R= ret.T

fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("SI Model")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# SEIS Model
def SEIS(y, t, N, beta, gamma, epsilon):
    S, E, I = y
    dSdt = -beta * S * I / N + gamma * I
    dEdt = beta * S * I / N - epsilon*E
    dIdt = epsilon*E - gamma * I
    return dSdt, dEdt, dIdt

N = 51740000
I0 = 1
S0 = N - I0
E0 = 3
y0 = S0, E0, I0

beta = 0.2161
gamma = 0.05
epsilon = 0.2

t = np.linspace(0, 240, 240)

ret = odeint(SEIS, y0, t, args=(N, beta, gamma, epsilon))
S, E, I = ret.T

fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E/N, 'purple', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
#ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("SEIS Model")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# SEIR Model
def SEIR(y, t, N, beta, gamma, epsilon):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - epsilon * E
    dIdt = epsilon*E-gamma*I
    dRdt= gamma*I
    return dSdt, dEdt, dIdt, dRdt

N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
E0 = 3
y0 = S0, E0, I0, R0

beta = 0.2161
gamma = 0.05
epsilon = 0.2

t = np.linspace(0, 240, 240)

ret = odeint(SEIR, y0, t, args=(N, beta, gamma, epsilon))
S, E, I, R = ret.T

fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E/N, 'purple', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("SI Model")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()



""" Parameter Study with SIR """
# Beta
N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
y0 = S0, I0, R0
gamma = 0.05
betas = np.array([])
Is = np.array([])
for i in range(100):
    beta = 0+1/100*i
    betas=np.append(betas, np.array([beta]), axis = 0)
    ret = odeint(SIR, y0, t, args=(N, beta, gamma))
    S, I, R= ret.T
    maxI = max(I)
    Is = np.append(Is, maxI)

fig = plt.figure(facecolor='w', figsize=(4,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(betas, Is, 'b', alpha=0.5, lw=2, label='maximum I')

ax.set_xlabel('Beta')
ax.set_ylabel('Maximum infected number')

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Maximum value of Class I depending on Beta")

legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# Gamma
N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
y0 = S0, I0, R0
beta = 0.2161
gammas = np.array([])
Is = np.array([])
for i in range(100):
    gamma = 0+1/100*i
    gammas=np.append(gammas, gamma)
    ret = odeint(SIR, y0, t, args=(N, beta, gamma))
    S, I, R= ret.T
    maxI = max(I)
    Is = np.append(Is, maxI)

fig = plt.figure(facecolor='w', figsize=(4,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(gammas, Is, 'b', alpha=0.5, lw=2, label='maximum I')

ax.set_xlabel('gamma')
ax.set_ylabel('Maximum infected number')

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Maximum value of Class I depending on Gamma")

legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# Finding Max and min of R0
betas = np.array([])
gammas = np.array([])
R0s = np.array([])
for i in range(100):
    beta = 0 + 1/100 * (i+1)
    
    for j in range(100):
        gamma = 0 + 1/100 *(j+1)
        
        betas = np.append(betas, beta)
        gammas = np.append(gammas, gamma)
        
        R0 = beta/gamma
        R0s = np.append(R0s, R0)

df = pd.DataFrame({'beta':betas, 'gamma':gammas, 'R0':R0s})
#display(df)

print("Max and Min of Reproduction index")
display(df[df['R0']== max(R0s)])
display(df[df['R0']== min(R0s)])


# Beta and Gamma
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot(betas, gammas, R0s, alpha=0.5)
ax.set_xlabel('beta')
ax.set_ylabel('gamma')
ax.set_zlabel('R0')
ax.set_title("Maximum value of Class I depending on Beta and Gamma")
ax.view_init(elev=30., azim=120)

plt.tight_layout()



# Beta = 0.2, Gamma = 0.05, R0 =4.0
N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
y0 = S0, I0, R0

beta = 0.2
gamma = 0.05
print("R0=", beta/gamma)

t = np.linspace(0, 500, 500)

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R= ret.T
print(S[30])
fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Beta = 0.2, Gamma = 0.05, R0 =4.0")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()



# Beta = 0.1, Gamma = 0.05, R0 =2.0
beta = 0.1
gamma = 0.05
print("R0=", beta/gamma)

t = np.linspace(0, 500, 500)

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R= ret.T
print(S[30])
fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Beta = 0.1, Gamma = 0.05, R0 =2.0")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()



# Beta = 0.08, Gamma = 0.05, R0 =1.6
beta = 0.08
gamma = 0.05
print("R0=", beta/gamma)

t = np.linspace(0, 500, 500)

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R= ret.T
print(S[30])
fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Beta = 0.08, Gamma = 0.05, R0 =1.6")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()



# Beta = 0.05, Gamma = 0.05, R0 =1
beta = 0.05
gamma = 0.05
print("R0=", beta/gamma)

t = np.linspace(0, 500, 500)

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R= ret.T
print(S[30])
fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Beta = 0.05, Gamma = 0.05, R0 =1")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# Beta = 0.2, Gamma = 0.125, R0 =1.6
beta = 0.2
gamma = 0.125
print("R0=", beta/gamma)

t = np.linspace(0, 500, 500)

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R= ret.T
print(S[30])
fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Beta = 0.2, Gamma = 0.125, R0 =1.6")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()



# Beta = 0.2, Gamma = 0.18, R0 =1.11
beta = 0.2
gamma = 0.18
print("R0=", beta/gamma)

t = np.linspace(0, 500, 500)

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R= ret.T
print(S[30])
fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title("Beta = 0.2, Gamma = 0.18, R0 =1.11")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()




""" Complex SIR Model """
Ss = np.array([])
Is = np.array([])
Rs = np.array([])

N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
y0 = S0, I0, R0

#       -30, 30-63, 63-108, 108-197, 197-224, 224-284
betas = [0.3, 0.4,  0.05,   0.1,     0.4,     0.01]
gamma = 0.05
xi = 0.2611 * 0.2161

# normal
t1 = np.linspace(0, 30, 30)
ret = odeint(SIR, y0, t1, args=(N , betas[0], gamma))
S1, I1, R1= ret.T

# shincheonji
t2 = np.linspace(30, 63, 33)
y0 = S1[29], I1[29], R1[29]
ret = odeint(SIR, y0, t2, args=(N, betas[1], gamma))
S2, I2, R2= ret.T
Ss = np.append(S1, S2[1:])
Is = np.append(I1, I2[1:])
Rs = np.append(R1, R2[1:])

# social distance stage1
t3 = np.linspace(63, 108, 45)
y0 = S2[32], I2[32], R2[32]
ret = odeint(SIR, y0, t3, args=(N, betas[2], gamma))
S3, I3, R3= ret.T
Ss = np.append(Ss, S3[1:])
Is = np.append(Is, I3[1:])
Rs = np.append(Rs, R3[1:])

# club
t4 = np.linspace(108, 197, 89)
y0 = S3[44], I3[44], R3[44]
ret = odeint(SIR, y0, t4, args=(N, betas[3], gamma))
S4, I4, R4= ret.T
Ss = np.append(Ss, S4[1:])
Is = np.append(Is, I4[1:])
Rs = np.append(Rs, R4[1:])

# church
t5 = np.linspace(197, 224, 27)
y0 = S4[88], I4[88], R4[88]
ret = odeint(SIR, y0, t5, args=(N, betas[4], gamma))
S5, I5, R5= ret.T
Ss = np.append(Ss, S5[1:])
Is = np.append(Is, I5[1:])
Rs = np.append(Rs, R5[1:])

# social distance
t6 = np.linspace(224, 284, 60)
y0 = S5[26], I5[26], R5[26]
ret = odeint(SIR, y0, t6, args=(N, betas[5], gamma))
S6, I6, R6= ret.T
Ss = np.append(Ss, S6[1:])
Is = np.append(Is, I6[1:])
Rs = np.append(Rs, R6[1:])


ts = np.linspace(0, 279, 279)



fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(ts, Ss/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(ts, Is/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(ts, Rs/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title(" Complex SIR Model ")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()



""" Complex SIR Model """
Ss = np.array([])
Is = np.array([])
Rs = np.array([])

N = 51740000
I0 = 1
R0 = 0
S0 = N - I0
y0 = S0, I0, R0

#       -30, 30-63, 63-108, 108-197, 197-224, 224-284
betas = [0.3, 0.4,  0.05,   0.1,     0.4,     0.01]
gamma = 0.05
xi = 0.2611 * 0.2161

# normal
t1 = np.linspace(0, 30, 30)
ret = odeint(SIRS, y0, t1, args=(N , betas[0], gamma, xi))
S1, I1, R1= ret.T

# shincheonji
t2 = np.linspace(30, 63, 33)
y0 = S1[29], I1[29], R1[29]
ret = odeint(SIRS, y0, t2, args=(N, betas[1], gamma, xi))
S2, I2, R2= ret.T
Ss = np.append(S1, S2[1:])
Is = np.append(I1, I2[1:])
Rs = np.append(R1, R2[1:])

# social distance stage1
t3 = np.linspace(63, 108, 45)
y0 = S2[32], I2[32], R2[32]
ret = odeint(SIRS, y0, t3, args=(N, betas[2], gamma, xi))
S3, I3, R3= ret.T
Ss = np.append(Ss, S3[1:])
Is = np.append(Is, I3[1:])
Rs = np.append(Rs, R3[1:])

# club
t4 = np.linspace(108, 197, 89)
y0 = S3[44], I3[44], R3[44]
ret = odeint(SIRS, y0, t4, args=(N, betas[3], gamma, xi))
S4, I4, R4= ret.T
Ss = np.append(Ss, S4[1:])
Is = np.append(Is, I4[1:])
Rs = np.append(Rs, R4[1:])

# church
t5 = np.linspace(197, 224, 27)
y0 = S4[88], I4[88], R4[88]
ret = odeint(SIRS, y0, t5, args=(N, betas[4], gamma, xi))
S5, I5, R5= ret.T
Ss = np.append(Ss, S5[1:])
Is = np.append(Is, I5[1:])
Rs = np.append(Rs, R5[1:])

# social distance
t6 = np.linspace(224, 284, 60)
y0 = S5[26], I5[26], R5[26]
ret = odeint(SIRS, y0, t6, args=(N, betas[5], gamma, xi))
S6, I6, R6= ret.T
Ss = np.append(Ss, S6[1:])
Is = np.append(Is, I6[1:])
Rs = np.append(Rs, R6[1:])


ts = np.linspace(0, 279, 279)


fig = plt.figure(facecolor='w',figsize=(8,4))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(ts, Ss/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(ts, Is/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(ts, Rs/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Days')
ax.set_ylabel('Relative Population')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_title(" Complex SIRS Model")
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()








