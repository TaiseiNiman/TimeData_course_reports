##### Exercise(7) sample
##### (c) G.Tanaka @ NITech　+ 改変

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse
from reservoirpy.observables import spectral_radius

rpy.verbosity(0)
rng = np.random.default_rng(0)

# import data (sunspot.year)
data_sunspot = sm.datasets.get_rdataset('sunspot.year')
sunspot = np.array(data_sunspot.data.value).reshape(-1,1)
month = np.array(data_sunspot.data.time).reshape(-1,1)
# normalize
sunspot_norm = (sunspot - sunspot.min()) / (sunspot.max() - sunspot.min())

# divide into training and test data
train_ratio = 0.9
horizon = 1
len_train = int(train_ratio*len(sunspot_norm))  # training period
len_test = len(sunspot)-len_train-horizon  # test period
u_train = sunspot_norm[:len_train]
d_train = sunspot_norm[horizon:len_train+horizon]
u_test = sunspot_norm[len_train:len_train+len_test] 
d_test = sunspot_norm[len_train+horizon:len_train+len_test+horizon]

# model parameters
Nu = 1  # num of input nodes
Nx = 200  # num of reservoir nodes
Ny = 1  # num of output nodes
density_Win = 1.0  # connection density of input-reservoir
density_W = 0.1  # connection density of reservoir-output
density_Wfb = 1.0  # connection density of output-reservoir
input_scaling = 1.0  # input scaling
sr = 0.5  # spectral radius
lr = 1  # leaking rate

# input weight matrix
Win = rng.random((Nx, Nu)) - 0.5
mask = rng.random((Nx, Win.shape[1]))
Win[mask > density_Win] = 0
Win = Win * input_scaling

# recurrent weight matrix
W = rng.random((Nx, Nx)) - 0.5
mask = rng.random((Nx, Nx))  # create a mask Uniform[0;1]
W[mask > density_W] = 0  # set to zero some connections
original_spectral_radius = spectral_radius(W)
W = W * (sr/original_spectral_radius)

# feedback weight matrix
Wfb = rng.random((Nx, Ny)) - 0.5
mask = rng.random((Nx, Wfb.shape[1]))
Wfb[mask > density_Wfb] = 0

def reservoir_lerning_settings(Nx, lr=1.0, sr=0.5, input_scaling=1.0, \
                      density_W=0.1, density_Win=1.0, density_Wfb=1.0):  
       
    # create a reservoir and a readout
    reservoir = Reservoir(Nx, lr=lr, sr=sr, input_scaling=input_scaling, \
                        rc_connectivity=density_W, input_connectivity=density_Win, \
                        fb_connectivity=density_Wfb, seed=0)
    readout = Ridge(ridge=1e-5)

    # training
    esn = reservoir >> readout
    feedback = 'TRUE'
    if feedback:
            reservoir = reservoir << readout
    reservoir.reset()

    return esn

def reservoir_pred(u_train,u_test,esn):
    len_train = len(u_train)
    len_test = len(u_test)
    # one-step-ahead prediction
    len_warming = 10
    warming_in = u_train[:len_warming]
    warming_out = esn.run(warming_in, reset=True)  # warmup
    y_pred_train = esn.run(u_train)
    y_pred_test = esn.run(u_test)

    # autoregressive prediction
    warming_in = u_train[len_train-len_warming:len_train]
    warming_out = esn.run(warming_in, reset=True)  # warmup
    y_pred_test_ar = np.zeros((len_test, 1))
    y = warming_out[-1]
    for t in range(len_test):  # generation
        y = esn(y)
        y_pred_test_ar[t, :] = y

    return y_pred_train, y_pred_test, y_pred_test_ar

sr_list = np.linspace(0.0, 1.2, 50)
rmse_list_onestep = []
rmse_list_ar = []
for sr in sr_list:
    esn = reservoir_lerning_settings(Nx, lr=lr, sr=sr, input_scaling=input_scaling, \
                            density_W=density_W, density_Win=density_Win, \
                            density_Wfb=density_Wfb)
    esn = esn.fit(u_train, d_train, warmup=10)  # train the model
    y_pred_train, y_pred_test, y_pred_test_ar = reservoir_pred(u_train, u_test, esn)
    # evluate error
    rmse_list_onestep.append(rmse(d_test, y_pred_test))
    print('RMSE (one-step-ahead): ', rmse_list_onestep[-1])
    rmse_list_ar.append(rmse(d_test, y_pred_test_ar))
    print('RMSE (autoregressive): ', rmse_list_ar[-1])

#print optimal spectral radius
optimal_sr_ar = sr_list[np.argmin(rmse_list_ar)]
optimal_sr_onestep = sr_list[np.argmin(rmse_list_onestep)]
print(f'Optimal spectral radius (one-step-ahead): {optimal_sr_onestep} in rmse = {rmse_list_onestep[np.argmin(rmse_list_onestep)]}')
print(f'Optimal spectral radius (ar): {optimal_sr_ar} in rmse = {rmse_list_ar[np.argmin(rmse_list_ar)]}')

#plot RMSE
plt.plot(sr_list, rmse_list_onestep, label='One-step-ahead', color='C3')
plt.plot(sr_list, rmse_list_ar, label='Autoregressive', color='C8')
plt.xlabel('Spectral Radius')
plt.ylabel('RMSE')
plt.title('the RMSE (Test Data) against the Spectral Radius')
plt.legend()
plt.show()

# train the model with the optimal spectral radius
esn = reservoir_lerning_settings(Nx, lr=lr, sr=optimal_sr_onestep, input_scaling=input_scaling, \
                            density_W=density_W, density_Win=density_Win, \
                            density_Wfb=density_Wfb)
esn = esn.fit(u_train, d_train, warmup=10)  # train the model
y_pred_train, y_pred_test, y_pred_test_ar = reservoir_pred(u_train, u_test, esn)

# transform to the original scale
y_pred_train = y_pred_train*(sunspot.max() - sunspot.min())+sunspot.min()
y_pred_test = y_pred_test*(sunspot.max() - sunspot.min())+sunspot.min()
y_pred_test_ar = y_pred_test_ar*(sunspot.max() - sunspot.min())+sunspot.min()

# plot
plt.plot(sunspot, color='gray', label='Data')
plt.plot(np.arange(0,len_train), y_pred_train, color='C0', label='Prediction (train)')
plt.plot(np.arange(len_train,len_train+len_test), y_pred_test, color='C3', label='Prediction (test, one-step-ahead)')
plt.plot(np.arange(len_train,len_train+len_test), y_pred_test_ar, color='C8', label='Prediction (test, autoregressive)')
plt.title(f'Sunspot (yearly) with optimal spectral radius (sr={optimal_sr_onestep.round(3)})')
plt.xlabel('$n$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.show()

esn = reservoir_lerning_settings(Nx, lr=lr, sr=optimal_sr_ar, input_scaling=input_scaling, \
                            density_W=density_W, density_Win=density_Win, \
                            density_Wfb=density_Wfb)
esn = esn.fit(u_train, d_train, warmup=10)  # train the model
y_pred_train, y_pred_test, y_pred_test_ar = reservoir_pred(u_train, u_test, esn)

# transform to the original scale
y_pred_train = y_pred_train*(sunspot.max() - sunspot.min())+sunspot.min()
y_pred_test = y_pred_test*(sunspot.max() - sunspot.min())+sunspot.min()
y_pred_test_ar = y_pred_test_ar*(sunspot.max() - sunspot.min())+sunspot.min()

# plot
plt.plot(sunspot, color='gray', label='Data')
plt.plot(np.arange(0,len_train), y_pred_train, color='C0', label='Prediction (train)')
plt.plot(np.arange(len_train,len_train+len_test), y_pred_test, color='C3', label='Prediction (test, one-step-ahead)')
plt.plot(np.arange(len_train,len_train+len_test), y_pred_test_ar, color='C8', label='Prediction (test, autoregressive)')
plt.title('Sunspot (yearly) with optimal spectral radius (ar={}) '.format(optimal_sr_ar.round(3)))
plt.xlabel('$n$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.show()