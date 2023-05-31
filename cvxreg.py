import numpy as np
from frozen import *
import pandas as pd
from collections import OrderedDict
import time
from joblib import Parallel, delayed, Memory

estimators = OrderedDict()
def get_estimator(estimator_name):
    return estimators[estimator_name]

from common.ols import OLSEstimator
estimators['OLS'] = OLSEstimator()

from algorithm.cnls.cnls import CNLSEstimator
from algorithm.amap.amap import AMAPEstimator
from algorithm.apcnls.apcnls import APCNLSEstimator

estimators["APCNLS"] = APCNLSEstimator(train_args={'use_L':True})
estimators["AMAP"] = AMAPEstimator(train_args={'use_L':True})
estimators['CNLS_star'] = CNLSEstimator(train_args={'use_L': True})
estimators['CNLS_ln'] = CNLSEstimator(train_args={'use_L': True, 'ln_L': True})

parallel_nworkers = 1  # maximum number of parallel workers (make sure you have enough RAM too)
parallel_backend = 'multiprocessing'
nruns = 1




 # domain dimensions
nsamples = (100, 250)  # number of samples
L = np.inf  # Lipschitz limit (can be set as a function to measure L on the union of the training and test sets)
L_scaler = 1.0  # multiplying L (makes sense when L is measured on the data)

def loss(yhat, y):  # L2-error
    return np.mean(np.square(yhat - y))

def fstar(X):
    return 0.5 * np.sum(np.square(X), axis=1)

def L_func(X):
    return max(np.linalg.norm(X, ord=2, axis=1))

L = L_func

covariate_std = 1.0

def sample_X(n, d):
    return np.random.randn(n, d) * covariate_std
env = gym.make("FrozenLake-v1",render_mode="ansi",desc=["SFFF", "FFFF", "FFFF", "FFFF"],is_slippery=False)
S = env.observation_space.n
A = env.action_space.n
domain_dims = (S*A,) 
#x,y = get_samples(env,10,1000)
#x_test,y_test = get_samples(env,1000,100)
def run_experiment(d, n, L, estimator_name, run,env):
        X,y = get_samples(env,5,n,rand_start=True)
        y_true = y
        #X_test = np.random.uniform (low=0, high=1, size= (10,64))
        #y_test = np.zeros(10)
        #for i in range(10):
        #     y_test[i] = entropy(X_test[i,])
        X_test,y_test = get_samples(env,5,100,rand_start=True)
        print(np.shape(X_test))
        #X, y, y_true, X_test, y_test = get_data(d, n, run, data_random_seed)
        L_true = max(L(X), L(X_test)) if callable(L) else L
        Lscaler = eval(L_scaler) if isinstance(L_scaler, str) else L_scaler
        L_est = (L_true * Lscaler) if np.isfinite(L_true) else np.inf

        X_norms = np.linalg.norm(X, axis=1)
        X_test_norms = np.linalg.norm(X_test, axis=1)
        result = OrderedDict()
        estimator = get_estimator(estimator_name)

        train_args = OrderedDict()
        if np.isfinite(L_est):
            train_args['L'] = L_est
        result['L_est'] = L_est
        result['L_true'] = L_true

        real_time, cpu_time = time.time(), time.perf_counter()
        model = estimator.train(X, y, **train_args)
        result['model'] = model
        result['nweights'] = model.weights.shape[0]
        result['max_weight_norm'] = max(np.linalg.norm(model.weights, axis=1))
        yhat = estimator.predict(model, X)
        result['train_risk'] = loss(yhat, y)
        result['train_err'] = loss(yhat, y_true)
        result['train_diff_mean'] = np.mean(yhat - y)
        result['train_diff_median'] = np.median(yhat - y)
        result['train_cpu_time'] = time.perf_counter() - cpu_time
        result['train_real_time'] = time.time() - real_time

        real_time, cpu_time = time.time(), time.perf_counter()
        yhat_test = estimator.predict(model, X_test)
        result['test_err'] = loss(yhat_test, y_test)
        print("test error")
        print(loss(yhat_test, y_test))
        result['test_cpu_time'] = time.perf_counter() - cpu_time
        result['test_real_time'] = time.time() - real_time
        return ((d, n, estimator_name, run), result)

def cached_func(func, estimator_name):
    return func
run_experiment(S*A,65,L,'OLS',1,env)
run_experiment(S*A,65,L,'APCNLS',1,env)
#results = []
#delayed_funcs = []
#for d in domain_dims:
#    for n in nsamples:
#        for estimator_name in estimators.keys():
#            for run in range(nruns):
#                print("in run")
#                delayed_funcs.append(delayed(cached_func(run_experiment, estimator_name))(
#                    d, n, L, estimator_name, run
#                ))
#print(delayed_funcs)
#Parallel(n_jobs=parallel_nworkers, backend=parallel_backend)(delayed_funcs)
#results = OrderedDict(sorted(Parallel(n_jobs=parallel_nworkers, backend=parallel_backend)(delayed_funcs)))
#print(delayed_funcs)
#print(results)
#results = OrderedDict(sorted(Parallel(n_jobs=parallel_nworkers, backend=parallel_backend)(delayed_funcs)))
#info('All results have been calculated.')
