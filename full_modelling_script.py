"""
Author: Jack Carlin
CID: 01353252
Title: Heat Dissipation in Microprocessors
"""
#%%
#### Cell for imported packages ###############################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.optimize import curve_fit

#%%
#### Creating the Microprocessor System and Modelling the Heat Motion ######### 
class grid:
    """
    Creating a class which will initialise the grid, identify points which 
    belong to certain components, and iterate over all points solving the heat
    equation.
    """
    def __init__(self, chip = [14,1], case = [20,2], conds = [1,1,1], res = 0.1, heatsink=False, convec = False):
        self.chip = chip
        self.case = case
        self.conds = conds
        self.res = res
        self.iterations = 0
        self.heatsink = heatsink
        self.convec = convec
        if convec:
            self.windspeed = convec
        if self.heatsink == False:
            dimensions = [case[0], (chip[1]+case[1])]  
            X, Y = np.ceil(dimensions[0]/self.res), np.ceil(dimensions[1]/self.res)
            self.X, self.Y = int(X)+1, int(Y)+1
            self.grid = np.ones([self.X+3, self.Y+3])*293.15
            self.update = np.ones([self.X+3, self.Y+3])*293.15 
            #Generating the indices for the points in the case and chip separately
            
            d = ((self.case[0] - self.chip[0])/(2*self.res))
            self.chip_x = [int(1 + d), int(self.X + (1 - d))]
            self.chip_y = [1 , int(self.chip[1]/self.res + 1)]  
            
            for i in range(self.chip_x[0], self.chip_x[1]):
                for j in range(self.chip_y[0], self.chip_y[1]):
                    self.update[i,j] = 500

            self.case_x = [1, self.X + 1]
            self.case_y = [int(1 + self.chip[1]/self.res), int(1 + (self.chip[1] + self.case[1])/self.res)]
    
            for i in range(self.case_x[0], self.case_x[1]):
                for j in range(self.case_y[0], self.case_y[1]):
                    self.update[i,j] = 500
                    
            self.boundary_updater()
            self.grid = self.update.copy()
                    
        else:
            #heatsink shape = [[base width, base height], [fin spacing, fin width, fin height]
            self.base = self.heatsink[0]
            self.fins = self.heatsink[1]
            dimensions = [self.base[0], (chip[1]+case[1]+self.base[1]+self.fins[2])]
            X, Y = np.ceil(dimensions[0]/self.res), np.ceil(dimensions[1]/self.res)
            self.X, self.Y = int(X)+1, int(Y)+1
            self.grid = np.ones([self.X+2,self.Y+2])*293.15
            self.update = np.ones([self.X+2, self.Y+2])*293.15
            
            d1 = ((self.base[0] - self.case[0])/(2*self.res))
            d2 = ((self.case[0] - self.chip[0])/(2*self.res))
            
            self.fin_x = [1, int(self.fins[1]/self.res) + 2]
            self.fin_y = [1 + int((self.chip[1]+self.case[1]+self.base[1])/self.res), 1 + int((self.chip[1]+self.case[1]+self.base[1]+self.fins[2])/self.res)]
            self.finspace = [3 + int(self.fins[1]/self.res), int((self.fins[0]+self.fins[1])/self.res)]
            self.base_x = [1, self.X+1]
            self.base_y = [int(1+(self.chip[1]+self.case[1])/self.res), int(1+(self.chip[1]+self.case[1]+self.base[1])/self.res)]
            
            self.case_x = [int(1+d1), int(self.X + (1-d1))]
            self.case_y = [int(1+self.chip[1]/self.res), int(1+(self.chip[1]+self.case[1])/self.res)]
            
            self.chip_x = [int(1+d1+d2), int(self.X + (1-(d1+d2)))]
            self.chip_y = [1, 1 + int(self.chip[1]/self.res)]
            
            self.finlast_x = [(int((self.base_x[1] - (self.fin_x[1]-1)))), int((self.base_x[1]))]
            self.finlast_y = self.fin_y
            
            self.w = int((self.fins[0] + self.fins[1])/self.res)
            
            self.finperiod = int(np.floor(self.base[0]/(self.fins[0] + self.fins[1])))
            
            for i in range(self.chip_x[0], self.chip_x[1]):
                for j in range(self.chip_y[0], self.chip_y[1]):
                    self.update[i,j] = 300
                    
            for i in range(self.case_x[0], self.case_x[1]):
                for j in range(self.case_y[0], self.case_y[1]):
                    self.update[i,j] = 300
                    
            for i in range(self.base_x[0], self.base_x[1]):
                for j in range(self.base_y[0], self.base_y[1]):
                    self.update[i,j] = 300
                    
            for k in range(0,self.finperiod):
                for i in range(self.fin_x[0]+k*self.w,self.fin_x[1]+k*self.w):
                    for j in range(self.fin_y[0], self.fin_y[1]):
                        self.update[i,j] = 300
                        
            for i in range(self.finlast_x[0], self.finlast_x[1]):
                for j in range(self.finlast_y[0], self.finlast_y[1]):
                    self.update[i,j] = 300
                    
            self.boundary_updater()
            self.grid = self.update.copy()
                    
    
    
    def grid_display(self):
        output=np.zeros((self.X+13, self.Y+13))
        output[5:self.X+7, 5:self.Y+7] = self.grid
        return output
        
    def update_point(self, i, j):
        new_T = (self.grid[i[0]+1:i[1]+1,j[0]:j[1]] + self.grid[i[0]-1:i[1]-1,j[0]:j[1]] + self.grid[i[0]:i[1],j[0]+1:j[1]+1] + self.grid[i[0]:i[1],j[0]-1:j[1]-1])
        new_T /= 4
        if (i == self.chip_x) and (j == self.chip_y):
            new_T += ((self.res**2)/4) * 0.5/self.conds[0] 
        
        self.update[i[0]:i[1],j[0]:j[1]] = new_T
    
    def updater(self):
        if self.heatsink == False:
            
            self.update_point(self.chip_x, self.chip_y)
                
            self.update_point(self.case_x, self.case_y)
                        
        else:
            self.update_point(self.chip_x, self.chip_y)
                
            self.update_point(self.case_x, self.case_y)
                    
            self.update_point(self.base_x, self.base_y)
                   
            for k in range(0,self.finperiod):
                self.update_point(list(np.add(self.fin_x, k*self.w)), self.fin_y)
            """    
            for k in range(0,self.finperiod):
                i = [self.fin_x[0]+k*self.w, self.fin_x[1]+k*self.w]
                j = 
                self.update_point(
            """        
            self.update_point(self.finlast_x, self.finlast_y)
                    
            
                
    def boundary_points(self, i, j, d = None, k = 1):
        
        if d == 1:
            if not self.convec:
                h = abs(1.31*np.cbrt(self.update[i[0]:i[1],j-1]-293.15))*1e-6
            else:
                h = (11.4 + 5.7*self.windspeed)*1e-6
            dT = self.update[i[0]:i[1],j-1]-293.15
            self.update[i[0]:i[1],j] = self.update[i[0]:i[1],j-2] - 2*self.res*((h*dT)/k)
        if d == 2:
            if not self.convec:
                h = abs(1.31*np.cbrt(self.update[i-1,j[0]:j[1]]-293.15))*1e-6
            else:
                h = (11.4 + 5.7*self.windspeed)*1e-6
            dT = self.update[i-1,j[0]:j[1]]-293.15
            self.update[i,j[0]:j[1]] = self.update[i-2,j[0]:j[1]] - 2*self.res*((h*dT)/k)
        if d == 3:
            if not self.convec:
                h = abs(1.31*np.cbrt(self.update[i[0]:i[1],j+1]-293.15))*1e-6
            else:
                h = (11.4 + 5.7*self.windspeed)*1e-6
            dT = self.update[i[0]:i[1],j+1]-293.15
            self.update[i[0]:i[1],j] = self.update[i[0]:i[1],j+2] - 2*self.res*((h*dT)/k)
        if d == 4:
            if not self.convec:
                h = abs(1.31*np.cbrt(self.update[i+1,j[0]:j[1]]-293.15))*1e-6
            else:
                h = (11.4 + 5.7*self.windspeed)*1e-6
            dT = self.update[i+1,j[0]:j[1]]-293.15
            self.update[i,j[0]:j[1]] = self.update[i+2,j[0]:j[1]] - 2*self.res*((h*dT)/k)
            
    def boundary_updater(self):
        if self.heatsink == False:

                
            self.boundary_points(self.case_x, self.case_y[1], 1, self.conds[1])

                
            self.boundary_points(self.case_x[1], self.case_y, 2, self.conds[1])

                
            l = [self.chip_x[1]+1, self.case_x[1]]
            self.boundary_points(l, self.case_y[0] - 1, 3, self.conds[1])

                
            self.boundary_points(self.chip_x[1], self.chip_y, 2, self.conds[0])
            
                
            self.boundary_points(self.chip_x, self.chip_y[0] - 1, 3, self.conds[0])
            
                
            self.boundary_points(self.chip_x[0] - 1, self.chip_y, 4, self.conds[0])
        
                
            l = [self.case_x[0], self.chip_x[0] -1]
            self.boundary_points(l, self.case_y[0] - 1, 3, self.conds[1])
            
                
            self.boundary_points(self.case_x[0] - 1, self.case_y, 4, self.conds[1])
                
        else:
                
            self.boundary_points(self.base_x[1], self.base_y, 2, self.conds[2]) 

            
            l = [self.case_x[1]+1, self.base_x[1]]
            self.boundary_points(l, self.base_y[0] - 1, 3, self.conds[2])

                
            self.boundary_points(self.case_x[1], self.case_y, 2, self.conds[1])
            
                
            l = [self.chip_x[1]+1, self.case_x[1]]
            self.boundary_points(l, self.case_y[0] - 1, 3, self.conds[1])

                
            self.boundary_points(self.chip_x[1], self.chip_y, 2, self.conds[0])

                
            self.boundary_points(self.chip_x, self.chip_y[0] - 1, 3, self.conds[0])

                
            self.boundary_points(self.chip_x[0] - 1, self.chip_y, 4, self.conds[0])

                
            l = [self.case_x[0], self.chip_x[0] -1]
            self.boundary_points(l, self.case_y[0] - 1, 3, self.conds[1])

                
            self.boundary_points(self.case_x[0] - 1, self.case_y, 4, self.conds[1])

                
            l = [self.base_x[0], self.case_x[0]-1]
            self.boundary_points(l, self.base_y[0]-1, 3, self.conds[2])

                
            self.boundary_points(self.base_x[0] - 1, self.base_y, 4, self.conds[2])
        
            for k in range(0,self.finperiod):
                    
                self.boundary_points(self.fin_x[0] + k*self.w - 1, self.fin_y, 4, self.conds[2])

                    
                self.boundary_points(np.add(self.fin_x, k*self.w), self.fin_y[1], 1, self.conds[2])
                    
                    
                self.boundary_points(self.fin_x[1] + k*self.w, self.fin_y, 2, self.conds[2])

                    
                self.boundary_points(np.add(self.finspace, k*self.w), self.fin_y[0], 1, self.conds[2])

                
            self.boundary_points(self.finlast_x[0]-1, self.fin_y, 4, self.conds[2])

                
            self.boundary_points(self.finlast_x, self.fin_y[1], 1, self.conds[2])

                
            self.boundary_points(self.finlast_x[1], self.fin_y, 2, self.conds[2])
            
            
            
    def update_T(self):
        self.updater()
        self.boundary_updater()
        
    def convergence(self):
        sigma1 = np.sum(self.update)
        sigma2 = np.sum(self.grid)
        c = (sigma1 - sigma2) / sigma2
        return c
    
    def convergence_(self):
        c = (np.max(self.update) - np.max(self.grid))/np.max(self.grid)
        return c
    
    def convergence_check(self):
        c = self.convergence()
        if abs(c) < 3e-8:
            return True, c
        else:
            return False, c
    
    def iterate(self):
        c_ = False
        print('Starting Iterations')
        """
        clist = []
        temps = []
        self.iterations=0
        while self.iterations < 1:
            self.update_T()
            c_, c__ = self.convergence_check()
            self.grid=self.update.copy()
            self.iterations+=1
            clist.append(c__)
            temps.append(np.max(self.grid))
            if self.iterations %1000 == 0:
                print(self.iterations)
                print(c__)  
        """
        clist = []
        temps = []
        while not c_: # or self.iterations < 3000000:
            self.update_T()
            c_, c__ = self.convergence_check()
            self.grid = self.update.copy()
            self.iterations += 1
            clist.append(c__)
            temps.append(np.max(self.grid))
            if self.iterations %1000 == 0:
                print(self.iterations)
                print(c__)
            
            if self.iterations == 3000000:
                break
               
        
        if c_ == True:
            print('Total Iterations:', self.iterations)
            
        return clist, temps
        
        
    
    def finheight_plotter(self, F_max, F_min, N):
        
        L = [F_min, F_max]
        T = []
        T_ = [0,0]
        finh = []
        for i in range(len(L)):
            g = grid(conds = [1.5e-1,2.3e-1,0.248], heatsink = [[31,3],[1,1,L[i]]],convec=20)
            g.iterate()
            G = g.grid_display()
            T.append(np.max(G))
            T_[i] = np.max[G]
            finh.append(L[i])
            print(T_)
        for n in range(N):
            F_mid = np.ceil((L[0] + L[1])/2)
            g = grid(conds = [1.5e-1,2.3e-1,0.248], heatsink = [[31,3],[1,1,F_mid]],convec=20)
            g.iterate()
            G = g.grid_display
            T.append(np.max(G))
            finh.append(F_mid)
            if T_[0] > T_[1]:
                L[0] = F_mid
                T_[0] = np.max(G)
            
            else:
                L[1] = F_mid
                T_[1] = np.max(G)
            
            
            
        plt.plot(finh, T)
        plt.show()
        #plt.savefig('/rds/general/user/jc9017/home/work/tempvheight.png')
        return finh, T
        

        #plt.savefig('/rds/general/user/jc9017/home/work/tempvheight.png')


#%%
g = grid(conds = [1.5e-1,2.3e-1,0.248], heatsink = [[61,4],[1,1,50]], convec=20)

G1 = g.grid

clist, temps = g.iterate()

G2 = g.grid

#%%
fig, ax=plt.subplots()
#ax.axhline(391, linestyle='--', color='k', alpha=0.3, linewidth=2)
ax.plot(np.linspace(1,g.iterations,g.iterations), temps, '-', color='crimson', lw=3)
ax.set_xlabel('Number of Iterations', fontsize=12)
ax.set_ylabel('Temperature / K', fontsize=12)
#ax.set_xlim(-100000, 5000000)
#ax.text(700000, 388, "390 Kelvin", color='k', ha='right', va='center', fontsize = 14, alpha=0.8)
ax.set_title('{} Iterations; Max Chip Temperature = {}K'.format(g.iterations,round(np.max(G2),-1)))
fig.show()

#%%
fig, ax=plt.subplots()
#ax.axhline(391, linestyle='--', color='k', alpha=0.3, linewidth=2)
ax.semilogx(clist, temps, '-', color='crimson', lw=3)
ax.set_xlim(np.max(clist), np.min(clist))
ax.set_xlabel('Convergence Parameter', fontsize=12)
ax.set_ylabel('Temperature / K', fontsize=12)
#ax.text(700000, 388, "390 Kelvin", color='k', ha='right', va='center', fontsize = 14, alpha=0.8)
ax.set_title('{} Iterations; Max Chip Temperature = {}K'.format(g.iterations,round(np.max(G2),-1)))
fig.show()

#%%
fig, ax=plt.subplots()
#ax.axhline(391, linestyle='--', color='k', alpha=0.3, linewidth=2)
ax.semilogy(np.linspace(1,5000000,5000000), clist, '-', color='cornflowerblue', lw=3)
ax.set_xlabel('Number of Iterations', fontsize=12)
ax.set_ylabel('Convergence Parameter', fontsize=12)
#ax.text(700000, 388, "390 Kelvin", color='k', ha='right', va='center', fontsize = 14, alpha=0.8)
ax.set_title('{} Iterations; Max Chip Temperature = {}K'.format(g.iterations,round(np.max(G2),-1)))
fig.show()

#%%

import numpy.ma as ma

G2_ = np.where(np.isnan(G2), 293.15, G2)
G2_ = ma.masked_invalid(G2_)
G2_ = ma.masked_less(G2_, 293.151)
print('Mean Chip Temperature:',np.mean(G2[g.chip_x[0]:g.chip_x[1],g.chip_y[0]:g.chip_y[1]]))
h = np.linspace(1,G2.shape[1],G2.shape[1])
w = np.linspace(1,G2.shape[0],G2.shape[0])
cm = plt.contourf(h, w, G2_, 30, cmap=plt.cm.inferno)
plt.colorbar(cm, label='Temperature in Kelvin')
plt.xlabel('Height / 0.1mm', fontsize=14)
plt.ylabel('Width / 0.1mm', fontsize=14)
plt.title('{} Iterations; Convection = 20m/s; Max Chip Temperature = {}K'.format(g.iterations,round(np.max(G2),-1)), fontsize=14)
plt.show()

#%%

def finheight_plotter(F_max, F_min, N):
        
    L = [F_min, F_max]
    T = []
    T_ = [0,0]
    finh = []
    for i in range(len(L)):
        g = grid(conds = [1.5e-1,2.3e-1,0.248], heatsink = [[61,4],[1,1,L[i]]])
        g.iterate()
        G = g.grid_display()
        T.append(np.max(G))
        T_[i] = np.max(G)
        finh.append(L[i])
    for n in range(N):
        F_mid = np.ceil((L[0] + L[1])/2)
        g = grid(conds = [1.5e-1,2.3e-1,0.248], heatsink = [[61,4],[1,1,F_mid]])
        g.iterate()
        G = g.grid_display()
        T.append(np.max(G))
        finh.append(F_mid)
        if T_[0] > T_[1]:
            L[0] = F_mid
            T_[0] = np.max(G)
        
        else:
            L[1] = F_mid
            T_[1] = np.max(G)
            
            
    print(finh)
    print(T)
    print(T_)
    plt.plot(finh, T)
    plt.show()
    #plt.savefig('/rds/general/user/jc9017/home/work/tempvheight.png')
    return finh, T

finh, T = finheight_plotter(60,20,4)

#%%
def finnumber_plotter(finnumber):
    T = []
    for i in range(len(finnumber)):
        s = (61 - finnumber[i])/(finnumber[i] - 1)
        g = grid(conds = [1.5e-1,2.3e-1,0.248], heatsink = [[61,4],[s,1,40]])
        g.iterate()
        G = g.grid_display()
        T.append(np.max(G))
        print('Loop {} Complete'.format(i+1))
        
    return T

        
fn = [11, 13, 16, 21, 31]
T_fn = finnumber_plotter(fn)

#%%
def exponential_fit(x,a,b,c):
    return a*np.exp(b*(-x)) + c


fn_ = np.linspace(10,40,31)
params, cov = curve_fit(exponential_fit, fn, T_fn)
plt.plot(fn_, exponential_fit(np.array(fn_), params[0], params[1], params[2]), linestyle='-', color='cornflowerblue', alpha=0.5, label='Exponential Fit')
plt.plot(fn, T_fn, 'x', ms=8, color='crimson', label='Data')
plt.ylabel('Maximum Chip Temperature / K', fontsize=14)
plt.xlabel('Number of Fins', fontsize=14)
plt.legend()
plt.show()
#%%
        
def exponential_fit(x,a,b,c):
    return a*np.exp(b*(-x)) + c



p0 = [1000, 0.05, 450]
params, cov = curve_fit(exponential_fit, finh, T)
#plt.plot(np.sort(finh), exponential_fit(np.array(np.sort(finh)), params[0], params[1], params[2]))
plt.plot(finh, T, 'x', ms=8, color='forestgreen')
plt.xlabel('Fin Height / mm')
plt.ylabel('Max Chip Temperature / K')
plt.show()