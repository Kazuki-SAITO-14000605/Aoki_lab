#import
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import minimize

#parameter
ng = 1.452 #diffractive index of glass
lam = 852e-9 #wave length[m]
R = 75e-3 #curvature radius[m]
L = 5e-3 #length of cavity[m]
f = np.array([30,40,50,75,100,125,150,200,250,300,400,500])*1e-3 #focus length[m]
tube_length = 0.0508 #[m]
#candidate: 50.8, 63.5, 76.3, 88.9[mm]
adjuster = 0.033
plot_on = False

#stability condition of FilterCavity
def stabCal(L,R,n,lam):
    ML = np.array([[1,L],[0,1]])
    MR = np.array([[1,0],[-2/R,1]]) #曲面鏡での反射

    M = np.matmul(MR,ML)
    M = np.matmul(ML,M)

    A=M[0][0]
    B=M[0][1]
    C=M[1][0]
    D=M[1][1]

    R_cav = 2*B/(D-A)
    Omega_cav = np.sqrt(2*B*lam/(np.pi*n*np.sqrt(4-(A+D)**2)))
    return R_cav, Omega_cav

R_cav, Omega_cav = stabCal(L,R,ng,lam)

# beam waist size and position
zs = np.array([17,25,33,41,49,57,65,73,81,89,95,103,111,117,119])*25*1e-3 #[m]
ws = np.array([1339,1379,1424,1475,1555,1652,1728,1810,1909,1958,2066,2165,2253,2313,2363])*1e-6 #[m]

lam = 852.3e-9

def waist(z, z0, w0):
    z_R = np.pi*1*w0**2/lam #Rayleigh length
    return w0*np.sqrt(1 + ((z-z0)/z_R)**2)

z = np.linspace(-4500e-3, 4500e-3, 9001)

popt, pcov = curve_fit(waist, zs, ws)
waist_position, Omega0 = popt
print("rayleigh length:",np.pi*1*popt[1]**2/lam,"[m]")
print("waist position:",waist_position,"[m]")
print("beam waist:",Omega0,"[m]")

if plot_on:
    plt.figure(figsize=(8,4))
    plt.plot(z, waist(z, *popt))
    plt.plot(zs, ws, '.')
    plt.xlabel("length[m]")
    plt.ylabel("radius[m]")
    plt.show()

# RayTransferMatrix
def matrixCal(d,f1,f2,n):
    d1=d[0]
    d2=d[1]
    d3=d[2]
    Mf1 = np.array([[1,0],[-1/f1,1]])
    Mf2 = np.array([[1,0],[-1/f2,1]])
    Md1 = np.array([[1,d1],[0,1]])
    Md2 = np.array([[1,d2],[0,1]])
    Md3 = np.array([[1,d3],[0,1]])
    Mn = np.array([[1,0],[0,1/n]]) #自由空間からlensに入射する時の平面境界での屈折

    M=np.matmul(Mf1,Md1)
    M=np.matmul(Md2,M)
    M=np.matmul(Mf2,M)
    M=np.matmul(Md3,M)
    M=np.matmul(Mn,M)

    A=M[0][0]
    B=M[0][1]
    C=M[1][0]
    D=M[1][1]

    q_in = 1j*np.pi*1*Omega0**2/lam
    q_out = (A*q_in+B)/(C*q_in+D)
    Omega_dash = np.sqrt(q_out.imag*lam/np.pi*1)
    CurvatureRadius_dash = 1/q_out.real
    return Omega_dash, CurvatureRadius_dash

def objective(d,f1,f2,n):
    diameter_error = (matrixCal(d,f1,f2,n)[0]- Omega_cav)**2
    return diameter_error

def curvature_constraint(d,f1,f2,n):
    return 1/matrixCal(d,f1,f2,n)[1]

# d1,d2,d3の範囲, 初期値
bounds = [(0-waist_position, 0.5-waist_position), (tube_length, tube_length+adjuster), (0, 0.20)]
initial_guess = [(bounds[0][0]+bounds[0][1])/2, (bounds[1][0]+bounds[1][1])/2, (bounds[2][0]+bounds[2][1])/2]

final_list = []
for i in f: #f1
    for j in f: #f2
        #制約の定義
        constraints = {
        'type': 'eq', #等式制約
        'fun': curvature_constraint,
        'args': (i,j,ng)
        }
        result = minimize(objective, initial_guess, args=(i,j,ng), bounds=bounds, constraints=constraints)
        if result.success:
            optimized_d1, optimized_d2, optimized_d3 = result.x
            final_radius, final_curvature = matrixCal([optimized_d1,optimized_d2,optimized_d3], i, j, ng)
            final_list.append((final_radius, final_curvature, optimized_d1, optimized_d2, optimized_d3, i, j)) #[m]
            
if final_list:
    closest_radius = min(final_list, key=lambda x: abs(x[0] - Omega_cav))
    print("Omega cavity:", Omega_cav,"[m]")
    print(f"final radius: {closest_radius[0]}[m]")
    print(f"final cuevature radius:{closest_radius[1]}[m]")
    print(f"出射ポートからlens1までの距離: {(closest_radius[2]+waist_position) * 1e3} [mm], d2: {closest_radius[3] * 1e3} [mm], d3: {closest_radius[4] * 1e3} [mm]")
    print(f"focus length 1: {closest_radius[5] * 1e3} [mm], focus length 2: {closest_radius[6] * 1e3} [mm]")