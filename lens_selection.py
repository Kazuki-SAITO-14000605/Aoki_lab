# import
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping

# parameter
ng = 1.452 #diffractive index of glass
lam = 852e-9 #wave length[m]
R = 75e-3 #curvature radius[m]
L = 5e-3 #length of cavity[m]
f = np.array([-100,-75,-50,30,40,50,75,100,150,200,250,300,400,500])*1e-3 #focus length[m]
Omega_target = 0.5e-3 #目標ビーム半径[m]

d1_range_start = 0
d1_range_end = 400e-3
d2_range_start = 0
d2_range_end = 400e-3
d3_range_start = 100e-3
d3_range_end = 150e-3
sum_path = 700e-3 #[m]

z = np.linspace(-4500e-3, 4500e-3, 9001) #plot range
plot_on = False

# beam waist size and position
zs = np.array([10,12,14,16,18,20,22,24,26,28])*25*1e-3 #[m]
ws = np.array([1251,1402,1559,1695,1848,2110,2232,2470,2579,2847])/2*1e-6 #[m]

def waist(z, z0, w0):
    z_R = np.pi*1*w0**2/lam #Rayleigh length
    return w0*np.sqrt(1 + ((z-z0)/z_R)**2)

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
def matrixCal(d,f1,f2):
    d1=d[0]
    d2=d[1]
    d3=d[2]
    Mf1 = np.array([[1,0],[-1/f1,1]])
    Mf2 = np.array([[1,0],[-1/f2,1]])
    Md1 = np.array([[1,d1],[0,1]])
    Md2 = np.array([[1,d2],[0,1]])
    Md3 = np.array([[1,d3],[0,1]])

    M=np.matmul(Mf1,Md1)
    M=np.matmul(Md2,M)
    M=np.matmul(Mf2,M)
    M=np.matmul(Md3,M)

    A=M[0][0]
    B=M[0][1]
    C=M[1][0]
    D=M[1][1]

    q_in = 1j*np.pi*1*Omega0**2/lam
    q_out = (A*q_in+B)/(C*q_in+D)
    q_out_inverse = 1/q_out
    Omega_dash = np.sqrt(q_out.imag*lam/np.pi*1)
    #Omega_dash = np.sqrt(- lam/(np.pi*1*q_out_inverse.imag))
    R_dash = 1/q_out.real
    return Omega_dash, R_dash

# 目的関数
def objective(d,f1,f2):
    diameter_error = (matrixCal(d,f1,f2)[0]- Omega_target)**2
    return diameter_error

# 曲率半径に関する制約条件
def curvature_constraint(d,f1,f2):
    return 1/matrixCal(d,f1,f2)[1]

# Optical pathの長さに関する制約条件
def optical_path_constraint(d):
    return d[0]+d[1]+d[2] - sum_path

# d1,d2,d3のrange, initial values
bounds = [(d1_range_start-waist_position, d1_range_end-waist_position), (d2_range_start, d2_range_end), (d3_range_start, d3_range_end)]

initial_guesses = [
    [bounds[0][0], bounds[1][0], bounds[2][0]],
    [(bounds[0][0]+bounds[0][1])/2, (bounds[1][0]+bounds[1][1])/2, (bounds[2][0]+bounds[2][1])/2],
    [bounds[0][1], bounds[1][1], bounds[2][1]]
]

final_list = []
for i in f: #f1
    for j in f: #f2
        #制約の定義
        constraints = (
        {'type': 'eq', 'fun': curvature_constraint, 'args': (i,j)},
        {'type': 'eq', 'fun': optical_path_constraint}
        )
        
    # グローバル最適化を実行
        for initial_guess in initial_guesses:
            minimizer_kwargs = {
                'method': 'SLSQP',
                'args': (i, j),
                'bounds': bounds,
                'constraints': constraints
            }
            
            result = basinhopping(
                objective,
                initial_guess,
                minimizer_kwargs=minimizer_kwargs,
                niter = 10 # 反復回数は適宜調整
            )
            
            if result.lowest_optimization_result.success:
                optimized_d1, optimized_d2, optimized_d3 = result.x
                final_radius, final_curvature = matrixCal([optimized_d1, optimized_d2, optimized_d3], i, j)
                final_list.append((final_radius, final_curvature, optimized_d1, optimized_d2, optimized_d3, i, j))
            
if final_list:
    closest_radius = min(final_list, key=lambda x: abs(x[0] - Omega_target))
    print(f"final radius: {closest_radius[0]}[m]")
    print(f"final Rayleigh length: {np.pi*1*closest_radius[0]**2/lam}[m]")
    print(f"final cuevature radius:{closest_radius[1]}[m]")
    print(f"出射ポートからlens1までの距離: {(closest_radius[2]+waist_position) * 1e3} [mm], d2: {closest_radius[3] * 1e3} [mm], d3: {closest_radius[4] * 1e3} [mm]")
    print(f"focus length 1: {closest_radius[5] * 1e3} [mm], focus length 2: {closest_radius[6] * 1e3} [mm]")