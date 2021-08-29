"""
    This file contains the code used in the numerical examples in Section 6 
    of the paper. The accompanying file 'numerical_examples_output.py' 
    contains the output from running this script. 
"""

# Indicate whether a Nvidia graphics card is available and 
# the cudatoolkit package installed:
gpu_installed = True
# Indicate whether to execute all of the code, otherwise parts 
# of the code that take an excessive amount of time to execute
# will be skipped:
complete_version = True
import birdepy as bd
if gpu_installed:
    import birdepy.gpu_functions as bdg
import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import seaborn as sns
import tikzplotlib



# Begin a timer to see how long the entire script takes to run
overall_tic = time.time()

### SIMULATION EXPERIMENT ###
gamma = 0.75
nu = 0.25
alpha = 0.01
c = 1
param=[gamma, nu, alpha, c]
times = np.arange(0,101,1)
model = 'Hassell'
z0 = 10
K = 10**3

p_data_exact = bd.simulate.discrete(param, model, z0, times, k=K, seed=2021)
p_data_ea = bd.simulate.discrete(param, model, z0, times, k=K, method='ea', seed=2021)
p_data_ma = bd.simulate.discrete(param, model, z0, times, k=K, method='ma', seed=2021)
p_data_gwa = bd.simulate.discrete(param, model, z0, times, k=K, method='gwa', seed=2021)
if gpu_installed:
    p_data_gpu = bdg.discrete(param, model, z0, times[-1], k=K, seed=2021)



fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(times, p_data_exact[0:3, :].T, color='tab:blue')
axs.plot(times, np.mean(p_data_exact, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('exact')
tikzplotlib.save("exact_sim.tex")


fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(times, p_data_ea[0:3, :].T, color='tab:red')
axs.plot(times, np.mean(p_data_ea, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('ea')
tikzplotlib.save("ea_sim.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(times, p_data_ma[0:3, :].T, color='tab:green')
axs.plot(times, np.mean(p_data_ma, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('ma')
tikzplotlib.save("ma_sim.tex")

fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(times, p_data_gwa[0:3, :].T, color='tab:cyan')
axs.plot(times, np.mean(p_data_gwa, axis=0), color='k')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('gwa')
tikzplotlib.save("gwa_sim.tex")


times = [0, times[-1]]

tic = time.time()
p_data_exact = bd.simulate.discrete(param, model, z0, times, k=K, seed=2021)
toc_exact = time.time()-tic

tic = time.time()
p_data_ea = bd.simulate.discrete(param, model, z0, times, k=K, method='ea', seed=2021)
toc_ea = time.time()-tic

tic = time.time()
p_data_ma = bd.simulate.discrete(param, model, z0, times, k=K, method='ma', seed=2021)
toc_ma = time.time()-tic

tic = time.time()
p_data_gwa = bd.simulate.discrete(param, model, z0, times, k=K, method='gwa', seed=2021)
toc_gwa = time.time()-tic

if gpu_installed:
    tic = time.time()
    p_data_gpu = bdg.discrete(param, model, z0, times[-1], k=K, seed=2021)
    toc_gpu = time.time()-tic


fig, ax = plt.subplots(figsize=(5,2))
sns.kdeplot(p_data_exact[:, -1], color='tab:blue', label='exact')
sns.kdeplot(p_data_ea[:, -1], color='tab:red', label='ea', linestyle=(0, (3, 5, 1, 5, 1, 5)))
sns.kdeplot(p_data_ma[:, -1], color='tab:green', label='ma',  linestyle='-.')
sns.kdeplot(p_data_gwa[:, -1], color='tab:cyan', label='gwa',linestyle=':')
if gpu_installed:
    sns.kdeplot(p_data_gpu, color='tab:purple', label='gpu', linestyle=(0, (5,1)))
plt.legend(loc='upper right')
plt.xlim([125, 275])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
tikzplotlib.save("sim_kdes.tex")

print("Table: Simulation CPU times")

if gpu_installed:
    print(tabulate([["exact", "ea", "ma", "gwa", "gpu"],
              ["Compute time (secs)", toc_exact, toc_ea, toc_ma, toc_gwa, toc_gpu]],
             headers="firstrow", floatfmt=".4f", tablefmt='latex'))
else:
    print(tabulate([["exact", "ea", "ma", "gwa"],
          ["Compute time (secs)", toc_exact, toc_ea, toc_ma, toc_gwa]],
         headers="firstrow", floatfmt=".4f", tablefmt='latex'))

### CONTINUOUS SIMULATION EXPERIMENT ###
gamma = 0.5
nu = 0.45

times = np.arange(0,3,0.1)

dsc_sample_path = bd.simulate.discrete([gamma, nu], 'linear', 10, times, seed=2021)
cts_times, cts_sample_path = bd.simulate.continuous([gamma, nu], 'linear', 10, 3, seed=2021)

fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(10,2.5))
fig.suptitle("Linear simulated sample paths")

ax.plot(times, dsc_sample_path, '+', label='discrete')
ax.step(cts_times, cts_sample_path, label='continuous', where='post')
ax.legend(loc='upper left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
tikzplotlib.save("cts_dsc_sim.tex")

### VERHULST MODEL PROBABILITY EXPERIMENT 1 ###
alpha = 0.025
beta = 0
gamma = 0.8
nu = 0.4
param = [gamma, nu, alpha, beta]
t = 1
zz =  np.arange(0, 40, 1)
model = 'Verhulst'
z0 = 15

fig.suptitle("Small population Verhulst density approximations")

tic = time.time()
if gpu_installed:
    pp_sim = bdg.probability(z0, zz, t, param, model=model, k=10**6, seed=2021)
else:
    pp_sim = bd.probability(z0, zz, [t], param, model=model, method='sim', k=10**4, seed=2021)
toc_sim = time.time() - tic

tic = time.time()
pp_expm = bd.probability(z0, zz, t, param, model=model, method='expm')
toc_expm = time.time() - tic

fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_expm[0], '+')
axs.plot(zz, np.where((pp_expm[0]>=0)*(pp_expm[0]<=1), 1, 0.01), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('expm')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_expm.tex")

tic = time.time()
pp_uni = bd.probability(z0, zz, t, param, model=model, method='uniform', k=10**4)
toc_uni = time.time() - tic

fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_uni[0], '+')
axs.plot(zz, np.where((pp_uni[0]>=0)*(pp_uni[0]<=1), 1, 0.01), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('uniform')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_uniform.tex")

tic = time.time()
pp_erl = bd.probability(z0, zz, t, param, model=model, method='Erlang', k=10**3)
toc_erl = time.time() - tic

fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_erl[0], '+')
axs.plot(zz, np.where((pp_erl[0]>=0)*(pp_erl[0]<=1), 1, 0.01), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('Erlang')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_erlang.tex")

tic = time.time()
pp_ilt = bd.probability(z0, zz, t, param, model=model, method='ilt')
toc_ilt = time.time() - tic

fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_ilt[0], '+')
axs.plot(zz, np.where((pp_ilt[0]>=0)*(pp_ilt[0]<=1), 1, 0.01), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('ilt')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_ilt.tex")

tic = time.time()
pp_da = bd.probability(z0, zz, t, param, model=model, method='da')
toc_da = time.time() - tic


fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_da[0], '+')
axs.plot(zz, np.where((pp_da[0]>=0)*(pp_da[0]<=1), 1, 0.01), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('da')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_da.tex")

tic = time.time()
pp_oua = bd.probability(z0, zz, t, param, model=model, method='oua')
toc_oua = time.time() - tic


fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_oua[0], '+')
axs.plot(zz, np.where((pp_oua[0]>=0)*(pp_oua[0]<=1), 1, 0.01), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('oua')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_oua.tex")

tic = time.time()
pp_gwa = bd.probability(z0, zz, t, param, model=model, method='gwa')
toc_gwa = time.time() - tic


fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_gwa[0], '+')
axs.plot(zz, np.where((pp_gwa[0]>=0)*(pp_gwa[0]<=1), 1, 0.01), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('gwa')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_gwa.tex")

tic = time.time()
pp_gwasa = bd.probability(z0, zz, t, param, model=model, method='gwasa')
toc_gwasa = time.time() - tic

fig, axs = plt.subplots(figsize=(5,2))
axs.plot(zz, pp_sim[0], 'tab:gray',alpha=0.5)
axs.plot(zz, pp_gwasa[0], '+')
axs.plot(zz, np.where((pp_gwasa[0]>=0)*(pp_gwasa[0]<=1), 1, 0.14), 'rx')
axs.set_ylim([0,0.15])
axs.set_title('gwasa')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
tikzplotlib.save("prob_gwasa.tex")


print("Table: Small population Verhulst CPU times")

print(tabulate([["sim", "expm", "uniform", "Erlang", "ilt", "da", "oua", "gwa", "gwasa"],
          ["Compute time (secs)", toc_sim, toc_expm, toc_uni, toc_erl, toc_ilt, toc_da, toc_oua, toc_gwa, toc_gwasa]],
         headers="firstrow", floatfmt=".4f", tablefmt='latex'))



### ESTIMATION EXPERIMENT 1 SETUP ###
gamma = 0.8
nu = 0.4
alpha = 0.025
beta = 0
param = [gamma, nu, alpha, beta]
num_sample_paths = 5

obs_times = [t for t in range(100)]
p_data = bd.simulate.discrete(param, 'Verhulst', 5, obs_times, seed=2021, k=num_sample_paths)
t_data = [obs_times for _ in range(num_sample_paths)]

con = {'type': 'ineq', 'fun': lambda p: p[0]-p[1]}
alpha_max = 1/np.amax(np.array(p_data))
alpha_mid = 0.5*alpha_max

if complete_version:
    ### ABC ESTIMATION EXPERIMENT ###
    est = bd.estimate(t_data, p_data, [0.5], [[0,1]], framework='abc', model='Verhulst',
                      known_p=[nu, alpha, beta], idx_known_p=[1, 2, 3], 
                      seed=2021, display=False)

    print('ABC estimate is', est.p, ', with standard error', est.se,
          'computed in ', est.compute_time, 'seconds.')

# ### DNM ESTIMATION EXPERIMENT ###
dnm_estimates = []
dnm_se = []
dnm_times = []

dnm_estimates.append(['gamma', 'nu', 'alpha'])
dnm_se.append(['gamma', 'nu', 'alpha'])
dnm_times.append(["Method", "Compute time (secs)"])

if complete_version:
    for likelihood in ['da', 'Erlang', 'expm', 'gwa', 'gwasa', 'ilt', 'oua', 'uniform']:
        tic = time.time()
        est = bd.estimate(t_data, p_data, [0.51, 0.5, alpha_mid], [[1e-6,5], [1e-6,5], [1e-6, alpha_max]],
                          model='Verhulst', framework='dnm', known_p=[0], idx_known_p=[3], 
                          con=con, likelihood=likelihood, opt_method='differential-evolution')
        toc = time.time()-tic
        (est.p).insert(0, likelihood)
        (est.se).insert(0, likelihood)
        dnm_estimates.append(est.p)
        dnm_se.append(est.se)
        dnm_times.append([likelihood, toc])
else:
    for likelihood in ['da', 'Erlang', 'expm', 'gwa', 'gwasa', 'oua', 'uniform']:
        tic = time.time()
        est = bd.estimate(t_data, p_data, [0.51, 0.5, alpha_mid], [[1e-6,5], [1e-6,5], [1e-6, alpha_max]],
                          model='Verhulst', framework='dnm', known_p=[0], idx_known_p=[3], 
                          con=con, likelihood=likelihood, opt_method='differential-evolution')
        toc = time.time()-tic
        (est.p).insert(0, likelihood)
        (est.se).insert(0, likelihood)
        dnm_estimates.append(est.p)
        dnm_se.append(est.se)
        dnm_times.append([likelihood, toc])   

print("Table: DNM Estimates")
print(tabulate(dnm_estimates, headers="firstrow", floatfmt=".4f", tablefmt='latex'))
print("Table: DNM Standard Errors")
print(tabulate(dnm_se, headers="firstrow", floatfmt=".4f", tablefmt='latex'))
print("Table: DNM Compute Times")
print(tabulate(dnm_times, headers="firstrow", floatfmt=".4f", tablefmt='latex'))



### EM ESTIMATION EXPERIMENT ###
em_estimates = []
em_se = []
em_times = []

em_estimates.append(['gamma', 'nu', 'alpha'])
em_se.append(['gamma', 'nu', 'alpha'])
em_times.append(["Method", "Compute time (secs)"])

if complete_version: 
    for technique in ['expm', 'ilt', 'num']:
        for accelerator in ['cg', 'none', 'Lange', 'qn1', 'qn2']:
            tic = time.time()
            est = bd.estimate(t_data, p_data, [0.51, 0.5, alpha_mid], [[1e-6,5], [1e-6,5], [1e-6, alpha_max]],
                              framework='em', technique=technique, accelerator=accelerator,
                              model='Verhulst', known_p=[0], idx_known_p=[3], con=con, display=False)
            toc = time.time() - tic
            (est.p).insert(0, accelerator)
            (est.p).insert(0, technique)
            (est.se).insert(0, accelerator)
            (est.se).insert(0, technique)
            em_estimates.append(est.p)
            em_se.append(est.se)
            em_times.append([technique, accelerator, toc])
else:
    for technique in ['expm', 'num']:
        for accelerator in ['cg', 'none', 'Lange', 'qn1', 'qn2']:
            tic = time.time()
            est = bd.estimate(t_data, p_data, [0.51, 0.5, alpha_mid], [[1e-6,5], [1e-6,5], [1e-6, alpha_max]],
                              framework='em', technique=technique, accelerator=accelerator,
                              model='Verhulst', known_p=[0], idx_known_p=[3], con=con, display=False)
            toc = time.time() - tic
            (est.p).insert(0, accelerator)
            (est.p).insert(0, technique)
            (est.se).insert(0, accelerator)
            (est.se).insert(0, technique)
            em_estimates.append(est.p)
            em_se.append(est.se)
            em_times.append([technique, accelerator, toc])

print("Table: EM Estimates")
print(tabulate(em_estimates, headers="firstrow", floatfmt=".4f", tablefmt='latex'))
print("Table: EM Standard Errors")
print(tabulate(em_se, headers="firstrow", floatfmt=".4f", tablefmt='latex'))
print("Table: EM Compute Times")
print(tabulate(em_times, headers="firstrow", floatfmt=".4f", tablefmt='latex'))


### LSE ESTIMATION EXPERIMENT ###
lse_estimates = []
lse_se = []
lse_times = []

lse_estimates.append(['gamma', 'nu', 'alpha'])
lse_se.append(['gamma', 'nu', 'alpha'])
lse_times.append(["Method", "Compute time (secs)"])

if complete_version: 
    for squares in ['expm', 'fm', 'gwa']:
        tic = time.time()
        est = bd.estimate(t_data, p_data, [0.51, 0.5, alpha_mid], [[1e-6,1], [1e-6,1], [1e-6, alpha_max]],
                          framework='lse', model='Verhulst', idx_known_p=[3], known_p=[0], con=con,
                          squares=squares, se_type='simulated')
        toc = time.time() - tic
        (est.p).insert(0, squares)
        (est.se).insert(0, squares)
        lse_estimates.append(est.p)
        lse_se.append(est.se)
        lse_times.append([squares, toc])
else:
    for squares in ['expm', 'gwa']:
        tic = time.time()
        est = bd.estimate(t_data, p_data, [0.51, 0.5, alpha_mid], [[1e-6,1], [1e-6,1], [1e-6, alpha_max]],
                          framework='lse', model='Verhulst', idx_known_p=[3], known_p=[0], con=con,
                          squares=squares, se_type='simulated')
        toc = time.time() - tic
        (est.p).insert(0, squares)
        (est.se).insert(0, squares)
        lse_estimates.append(est.p)
        lse_se.append(est.se)
        lse_times.append([squares, toc])

print("Table: LSE Estimates")
print(tabulate(lse_estimates, headers="firstrow", floatfmt=".4f", tablefmt='latex'))
print("Table: LSE Standard Errors")
print(tabulate(lse_se, headers="firstrow", floatfmt=".4f", tablefmt='latex'))
print("Table: LSE Compute Times")
print(tabulate(lse_times, headers="firstrow", floatfmt=".4f", tablefmt='latex'))

overall_time = time.time() - overall_tic
print('Overall script compute time:', overall_time)


# The output as displayed by running the complete version
# of the code: 

# ABC estimate is [0.756014561439248] , with standard error [0.05198535020776311] computed in  4444.180299282074 seconds.


# Table: DNM Estimates
# \begin{tabular}{lrrr}
# \hline
#           &   gamma &     nu &   alpha \\
# \hline
#   da      &  0.7817 & 0.3906 &  0.0252 \\
#   Erlang  &  0.7824 & 0.3873 &  0.0250 \\
#   expm    &  0.7810 & 0.3867 &  0.0250 \\
#   gwa     &  0.4111 & 0.3414 &  0.0075 \\
#   gwasa   &  0.4199 & 0.3494 &  0.0075 \\
#   ilt     &  0.7799 & 0.3859 &  0.0250 \\
#   oua     &  0.6528 & 0.3716 &  0.0229 \\
#   uniform &  0.7810 & 0.3867 &  0.0250 \\
# \hline
# \end{tabular}

# Table: DNM Standard Errors
# \begin{tabular}{lrrr}
# \hline
#           &   gamma &     nu &   alpha \\
# \hline
#   da      &  0.0612 & 0.0292 &  0.0013 \\
#   Erlang  &  0.0653 & 0.0294 &  0.0013 \\
#   expm    &  0.0650 & 0.0292 &  0.0013 \\
#   gwa     &  0.0431 & 0.0228 &  0.0040 \\
#   gwasa   &  0.0437 & 0.0236 &  0.0040 \\
#   ilt     &  0.0645 & 0.0289 &  0.0013 \\
#   oua     &  0.0570 & 0.0298 &  0.0016 \\
#   uniform &  0.0650 & 0.0292 &  0.0013 \\
# \hline
# \end{tabular}

# Table: DNM Compute Times
# \begin{tabular}{lr}
# \hline
#  Method   &   Compute time (secs) \\
# \hline
#  da       &               78.1455 \\
#  Erlang   &                0.8940 \\
#  expm     &                1.1730 \\
#  gwa      &               96.3210 \\
#  gwasa    &                8.4930 \\
#  ilt      &             4654.8463 \\
#  oua      &               17.5510 \\
#  uniform  &                2.3520 \\
# \hline
# \end{tabular}

# Table: EM Estimates
# \begin{tabular}{llrrr}
# \hline
#       &       &   gamma &     nu &   alpha \\
# \hline
#  expm & cg    &  0.7796 & 0.3865 &  0.0250 \\
#  expm & none  &  0.8175 & 0.4079 &  0.0248 \\
#  expm & Lange &  0.8031 & 0.4039 &  0.0246 \\
#  expm & qn1   &  0.7810 & 0.3867 &  0.0250 \\
#  expm & qn2   &  0.7526 & 0.4068 &  0.0231 \\
#  ilt  & cg    &  0.7815 & 0.3862 &  0.0250 \\
#  ilt  & none  &  0.8171 & 0.4078 &  0.0248 \\
#  ilt  & Lange &  0.8060 & 0.4025 &  0.0248 \\
#  ilt  & qn1   &  0.7802 & 0.3864 &  0.0250 \\
#  ilt  & qn2   &  0.7535 & 0.4073 &  0.0238 \\
#  num  & cg    &  0.8626 & 0.4296 &  0.0249 \\
#  num  & none  &  0.6153 & 0.4414 &  0.0172 \\
#  num  & Lange &  0.7859 & 0.3895 &  0.0250 \\
#  num  & qn1   &  0.5046 & 0.5039 &  0.0161 \\
#  num  & qn2   &  0.6609 & 0.4275 &  0.0184 \\
# \hline
# \end{tabular}

# Table: EM Standard Errors
# \begin{tabular}{llrrr}
# \hline
#       &       &    gamma &       nu &    alpha \\
# \hline
#  expm & cg    &   0.0649 &   0.0292 &   0.0013 \\
#  expm & none  &   0.0712 &   0.0325 &   0.0013 \\
#  expm & Lange &   0.0700 &   0.0318 &   0.0013 \\
#  expm & qn1   &   0.0650 &   0.0292 &   0.0013 \\
#  expm & qn2   &   0.0655 &   0.0322 &   0.0015 \\
#  ilt  & cg    &   0.0649 &   0.0289 &   0.0012 \\
#  ilt  & none  &   0.0139 &   0.0138 &   0.0012 \\
#  ilt  & Lange &   0.0378 &   0.0305 & nan      \\
#  ilt  & qn1   &   0.0649 &   0.0290 &   0.0013 \\
#  ilt  & qn2   &   0.0592 &   0.0326 &   0.0013 \\
#  num  & cg    &   0.0778 &   0.0363 &   0.0012 \\
#  num  & none  &   0.0490 &   0.0423 &   0.0018 \\
#  num  & Lange &   0.0656 &   0.0296 &   0.0013 \\
#  num  & qn1   & nan      & nan      & nan      \\
#  num  & qn2   &   0.0625 &   0.0352 &   0.0020 \\
# \hline
# \end{tabular}

# Table: EM Compute Times
# \begin{tabular}{llr}
# \hline
#       & Method   &   Compute time (secs) \\
# \hline
#  expm & cg       &                4.9390 \\
#  expm & none     &                5.1770 \\
#  expm & Lange    &                4.9270 \\
#  expm & qn1      &               11.9160 \\
#  expm & qn2      &                4.1410 \\
#  ilt  & cg       &             9639.8409 \\
#  ilt  & none     &             8578.1230 \\
#  ilt  & Lange    &             9859.6772 \\
#  ilt  & qn1      &            21542.1499 \\
#  ilt  & qn2      &             6472.3525 \\
#  num  & cg       &               12.9565 \\
#  num  & none     &               89.3902 \\
#  num  & Lange    &               28.2298 \\
#  num  & qn1      &               47.7273 \\
#  num  & qn2      &               10.1772 \\
# \hline
# \end{tabular}

# Table: LSE Estimates
# \begin{tabular}{lrrr}
# \hline
#       &   gamma &     nu &   alpha \\
# \hline
#  expm &  0.7078 & 0.2988 &  0.0288 \\
#  fm   &  0.7109 & 0.2955 &  0.0279 \\
#  gwa  &  0.6792 & 0.3278 &  0.0244 \\
# \hline
# \end{tabular}

# Table: LSE Standard Errors
# \begin{tabular}{lrrr}
# \hline
#       &   gamma &     nu &   alpha \\
# \hline
#  expm &  0.0446 & 0.0366 &  0.0021 \\
#  fm   &  0.0365 & 0.0262 &  0.0025 \\
#  gwa  &  0.0506 & 0.0355 &  0.0021 \\
# \hline
# \end{tabular}
# Table: LSE Compute Times
# \begin{tabular}{lr}
# \hline
#  Method   &   Compute time (secs) \\
# \hline
#  expm     &               28.0345 \\
#  fm       &              496.2759 \\
#  gwa      &               18.6061 \\
# \hline
# \end{tabular}

# Overall script compute time: 71020.70624732971
