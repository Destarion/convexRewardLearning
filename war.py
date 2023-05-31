import gymnasium as gym
import numpy as np
from warenv import WarEnv
#from frozen import get_samples,choose_start
import random
from scipy.stats import entropy
from war_sample import get_samples_war
from cvxreg import *
#env = gym.make("FrozenLake-v1",render_mode="human",desc=["SFFF"],is_slippery=False)
env = WarEnv(render_mode="ansi",is_slippery=False)
env.reset()
print(env)
#print(get_samples_war(env,5,1)[1])

def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)

def run_experiment_war(d, n, L, estimator_name, run,env):
        X,y = get_samples_war(env,10,n,rand_start=True)
        y_true = y
        y_test = np.zeros(10)
        X_test = np.zeros((10,6))
        #for i in range(10):
        #     X_test[i,] = runif_in_simplex(6)
        #     print(sum(X_test[i,]))
        #     y_test[i] = -10*(np.sqrt(X_test[i,0]+X_test[i,1]) + np.sqrt(X_test[i,4]+X_test[i,5]))

        #print(X_test)
        X_test,y_test = get_samples_war(env,10,10)
        #print(np.shape(X_test))
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
run_experiment_war(S*A,50,L,'APCNLS',1,env)
run_experiment_war(S*A,50,L,'OLS',1,env)