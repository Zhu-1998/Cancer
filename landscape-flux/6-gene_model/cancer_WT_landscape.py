from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sys
import os
import time
from scipy.sparse import csr_matrix


##################################LHS###################################
def LHSample( D,bounds,N):

    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(low=j*d, high=(j+1)*d, size = 1)[0]           

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('error')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,(upper_bounds - lower_bounds),out=result),lower_bounds,out=result)
    return result
#########################################################################################
			  
 
x_lim=[0, 1.5]
y_lim=[0, 1.5]
Dim = 6    
N = 400
bounds = [[0,1.5],[0,1.5],[0,1.5],[0,1.5],[0,1.5],[0,1.5]]
LHS_of_paras = LHSample(Dim, bounds, N)

numTimeSteps=50000000
starttime = 10000000
Tra_grid = 200


sa=0.8
b=0.1
a1=0.5
a2=0.5
sa2=sa
sa1=sa
a5=0.5
a6=0.5
S1=0.5
S2=0.5
S3=0.5
S4=0.5
S5=0.5
S6=0.5
S7=0.5
K1=0.5
K2=0.5
K3=0.5
K4=0.5
K5=0.5
K6=0.5
K7=0.5
K8=0.5
K9=0.5
n=4.0
g=1.0

def path_function(D, dt, i):    
#    x_path = []
#    y_path = []
    num_tra = np.zeros((Tra_grid, Tra_grid))
    total_Fx = np.zeros((Tra_grid, Tra_grid))
    total_Fy = np.zeros((Tra_grid, Tra_grid))

    init_xy = LHS_of_paras[i, :]
    x0 = init_xy[0]
    x1 = init_xy[1]
    x2 = init_xy[2]
    x3 = init_xy[3]
    x4 = init_xy[4]
    x5 = init_xy[5]
 

    # Evaluate potential (Integrate) over trajectory from init cond to  stable steady state
    for n_steps in np.arange(1, numTimeSteps):
        # update dxdt, dydt
        F0 = sa1*x0**n/(S4**n+x0**n)+b*K4**n/(K4**n+x4**n)-g*x0
        F1 = sa2*x1**n/(S3**n+x1**n)+b*K2**n/(K2**n+x2**n)+b*K3**n/(K3**n+x3**n)-g*x1
        F2 = a1*x0**n/(S1**n+x0**n)+a2*x5**n/(S2**n+x5**n)+b*K1**n/(K1**n+x1**n)-g*x2
        F3 = a5*x0**n/(S5**n+x0**n)+b*K5**n/(K5**n+x1**n)+b*K6**n/(K6**n+x5**n)-g*x3
        F4 = a6*x0**n/(S6**n+x0**n)+b*K7**n/(K7**n+x3**n)-g*x4
        F5 = sa2*x5**n/(S7**n+x5**n)+b*K8**n/(K8**n+x0**n)+b*K9**n/(K9**n+x3**n)-g*x5
        
        # update x, y
        dx0 = F0 * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        dx1 = F1 * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        dx2 = F2 * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        dx3 = F3 * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        dx4 = F4 * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()
        dx5 = F5 * dt + np.sqrt(2*D) * np.sqrt(dt) * np.random.randn()     

        x0 = x0 + dx0
        x1 = x1 + dx1
        x2 = x2 + dx2
        x3 = x3 + dx3
        x4 = x4 + dx4
        x5 = x5 + dx5
        
        if x0 < x_lim[0]:
            x0 = 2 * x_lim[0] - x0
        if x1 < x_lim[0]:
            x1 = 2 * x_lim[0] - x1
        if x2 < x_lim[0]:
            x2 = 2 * x_lim[0] - x2
        if x3 < x_lim[0]:
            x3 = 2 * x_lim[0] - x3            
        if x4 < x_lim[0]:
            x4 = 2 * x_lim[0] - x4            
        if x5 < x_lim[0]:
            x5 = 2 * x_lim[0] - x5            
            
        if x0 > x_lim[1]:
            x0 = 2 * x_lim[1] - x0
        if x1 > x_lim[1]:
            x1 = 2 * x_lim[1] - x1
        if x2 > x_lim[1]:
            x2 = 2 * x_lim[1] - x2
        if x3 > x_lim[1]:
            x3 = 2 * x_lim[1] - x3
        if x4 > x_lim[1]:
            x4 = 2 * x_lim[1] - x4
        if x5 > x_lim[1]:
            x5 = 2 * x_lim[1] - x5

        F00 = sa1*x0**n/(S4**n+x0**n)+b*K4**n/(K4**n+x4**n)-g*x0
        F11 = sa2*x1**n/(S3**n+x1**n)+b*K2**n/(K2**n+x2**n)+b*K3**n/(K3**n+x3**n)-g*x1
        F22 = a1*x0**n/(S1**n+x0**n)+a2*x5**n/(S2**n+x5**n)+b*K1**n/(K1**n+x1**n)-g*x2
        F33 = a5*x0**n/(S5**n+x0**n)+b*K5**n/(K5**n+x1**n)+b*K6**n/(K6**n+x5**n)-g*x3
        F44 = a6*x0**n/(S6**n+x0**n)+b*K7**n/(K7**n+x3**n)-g*x4
        F55 = sa2*x5**n/(S7**n+x5**n)+b*K8**n/(K8**n+x0**n)+b*K9**n/(K9**n+x3**n)-g*x5

        
        if n_steps > starttime:
            A = int((x0 - x_lim[0]) * Tra_grid / (x_lim[1] - x_lim[0]))   
            B = int((x1 - y_lim[0]) * Tra_grid / (y_lim[1] - y_lim[0]))
            if A < Tra_grid and B<Tra_grid:
                num_tra[A, B] = num_tra[A, B] + 1;
                total_Fx[A, B] = total_Fx[A, B] + F00
                total_Fy[A, B] = total_Fy[A, B] + F11
        
    np.savetxt('num_tra_' + str(i) + '.csv', num_tra, delimiter=",") 
    np.savetxt('total_Fx_' + str(i) + '.csv', total_Fx, delimiter=",") 
    np.savetxt('total_Fy_' + str(i) + '.csv', total_Fy, delimiter=",") 
    

start = time.time()	
# 并行
Parallel(n_jobs=150)(
    delayed(path_function)(0.01, 0.0001, i)
    for i in range(N))

	
num_tra = np.zeros((Tra_grid, Tra_grid))
total_Fx = np.zeros((Tra_grid, Tra_grid))
total_Fy = np.zeros((Tra_grid, Tra_grid))

for i in range(N):
    num_tra_i = np.loadtxt(open('num_tra_' + str(i) + '.csv',"rb"),delimiter=",")
    total_Fx_i = np.loadtxt(open('total_Fx_' + str(i) + '.csv',"rb"),delimiter=",")
    total_Fy_i = np.loadtxt(open('total_Fy_' + str(i) + '.csv',"rb"),delimiter=",")
    num_tra = num_tra + num_tra_i
    total_Fx = total_Fx + total_Fx_i
    total_Fy = total_Fy + total_Fy_i
    
p_tra = num_tra / (sum(sum(num_tra)))
print([sum(sum(num_tra)), N*(numTimeSteps-starttime)])
pot_U = -np.log(p_tra)
mean_Fx = total_Fx / num_tra
mean_Fy = total_Fy / num_tra

xlin = np.linspace(x_lim[0], x_lim[1], Tra_grid)
ylin = np.linspace(y_lim[0], y_lim[1], Tra_grid)
Xgrid, Ygrid = np.meshgrid(xlin, ylin)

np.savetxt("num_tra.csv", num_tra, delimiter=",") 
np.savetxt("total_Fx.csv", total_Fx, delimiter=",") 
np.savetxt("total_Fy.csv", total_Fy, delimiter=",") 
np.savetxt("p_tra.csv", p_tra, delimiter=",") 
np.savetxt("pot_U.csv", pot_U, delimiter=",") 
np.savetxt("mean_Fx.csv", mean_Fx, delimiter=",") 
np.savetxt("mean_Fy.csv", mean_Fy, delimiter=",") 
np.savetxt("Xgrid.csv", Xgrid, delimiter=",") 
np.savetxt("Ygrid.csv", Ygrid, delimiter=",") 

Time = time.time() - start
print(str(Time) + 's')