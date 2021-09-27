"""
    This file contains the code used in the black robin and whooping crane case 
    studies in Section 5 of the paper. The accompanying file 
    'case_studies_output.py' contains the output from running this script. 
"""
import birdepy as bd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import tikzplotlib

### BLACK ROBIN DATASET 1 CASE STUDY ###
t_data_b = [1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 2010,
          2011, 2012, 2013, 2014, 2015]
p_data_b = [30, 37, 35, 35, 42, 39, 50, 50, 57, 61, 86, 98, 94, 108, 117, 118]


###############
est_v1 = bd.estimate(t_data_b, p_data_b, [2, 2, 0.05], [[0,10], [0,10], [0, 1]],
                    model='Verhulst',  known_p=[0], idx_known_p=[3],
                    opt_method='differential-evolution', seed=2021)
(est_v1.p).insert(0, 'Verhulst 1')
(est_v1.se).insert(0, 'Verhulst 1')
extra_v1 = ['Verhulst 1', est_v1.capacity[1], np.exp(est_v1.val)]
###############
est_v2 = bd.estimate(t_data_b, p_data_b, [2, 2, 0.05], [[0,10], [0,10], [0, 1]],
                    model='Verhulst',  known_p=[0], idx_known_p=[2], se_type='asymptotic',
                    opt_method='differential-evolution', seed=2021)
(est_v2.p).insert(0, 'Verhulst 2')
(est_v2.se).insert(0, 'Verhulst 2')
extra_v2 = ['Verhulst 2', est_v2.capacity[1], np.exp(est_v2.val)]
###############
est_r = bd.estimate(t_data_b, p_data_b, [2, 2, 0.05], [[0,10], [0,10], [0,1]], 
                    model='Ricker', known_p=[1], idx_known_p=[3], se_type='asymptotic', 
                    opt_method='differential-evolution', seed=2021)
(est_r.p).insert(0, 'Ricker')
(est_r.se).insert(0, 'Ricker')
extra_r = ['Ricker', est_r.capacity[1], np.exp(est_r.val)]
###############
est_b = bd.estimate(t_data_b, p_data_b, [2, 2, 2], [[0,10], [0,10], [0,1]],
                      model='Hassell', known_p=[1], idx_known_p=[3], 
                      se_type='asymptotic', opt_method='differential-evolution',
                      seed=2021)
(est_b.p).insert(0, 'Beverton-Holt')
(est_b.se).insert(0, 'Beverton-Holt')
extra_b = ['Beverton-Holt', est_b.capacity[1], np.exp(est_b.val)]
###############
est_h = bd.estimate(t_data_b, p_data_b, [2, 2, 2], [[0,10], [0,10], [0, 1]],
                      model='Hassell', known_p=[2], idx_known_p=[3], 
                      se_type='asymptotic', opt_method='differential-evolution',
                      seed=2021)
(est_h.p).insert(0, 'Hassell')
(est_h.se).insert(0, 'Hassell')
extra_h = ['Hassell', est_h.capacity[1], np.exp(est_h.val)]
###############
est_mss = bd.estimate(t_data_b, p_data_b, [2, 2, 2], [[0,10], [0,10], [0, 1]], 
                      model='MS-S', known_p=[2], idx_known_p=[3], 
                      se_type='asymptotic', opt_method='differential-evolution', 
                      seed=2021)
(est_mss.p).insert(0, 'MS-S')
(est_mss.se).insert(0, 'MS-S')
extra_mss = ['MS-S', est_mss.capacity[1], np.exp(est_mss.val)]
###############
est_l = bd.estimate(t_data_b, p_data_b, [2]*2, [[0,10]]*2, model='linear', 
                    se_type='asymptotic', opt_method='differential-evolution', 
                    seed=2021)
(est_l.p).insert(0, 'linear')
(est_l.se).insert(0, 'linear')
extra_l = ['linear', 'n/a', np.exp(est_l.val)]
###############
###############
###############
print("\n Table: BLACK ROBIN DATASET PARAMETER ESTIMATES")
print(tabulate([["Model", "gamma", "nu", "alpha/beta"],
            est_v1.p, est_v2.p, est_r.p, est_b.p, est_h.p, est_mss.p, est_l.p],
          headers="firstrow", floatfmt=".4f", tablefmt='latex'))
###############
print("Table: BLACK ROBIN DATASET STANDARD ERRORS")
print(tabulate([["Model", "gamma", "nu", "alpha", "beta/c"],
            est_v1.se, est_v2.se, est_r.se, est_b.se, est_h.se, est_mss.se, est_l.se],
          headers="firstrow", floatfmt=".4f", tablefmt='latex'))
###############
print("Table: BLACK ROBIN DATASET CARRYING CAPACITY AND LIKELIHOOD")
print(tabulate([["Model", "Capacity", "Likelihood"],
            extra_v1, extra_v2, extra_r, extra_b, extra_h, extra_mss, extra_l],
          headers="firstrow", tablefmt='latex'))
###############
###############

bd.forecast('Hassell', p_data_b[-1], np.arange(2015, 2051, 1), est_b.p[1:], cov=est_b.cov, p_bounds=[[0,10], [0,10], [0, 1]], 
            known_p=[1], idx_known_p=[3], export='hassell_robins')

bd.estimate(t_data_b, p_data_b, [0.36, 0.0017], [[0,1], [0, 1]],
            model='Hassell', known_p=[0.2373, 1], idx_known_p=[1, 3], 
            se_type='asymptotic', ci_plot=True, seed=2021, export='hassell_robins_ci_asy',
            xlabel='$\gamma$', ylabel='$\\alpha$')

bd.estimate(t_data_b, p_data_b, [0.36, 0.0017], [[0,10], [0, 1]],
            model='Hassell', known_p=[0.2373, 1], idx_known_p=[1, 3], 
            se_type='simulated', ci_plot=True, seed=2021, export='hassell_robins_ci_sim',
            xlabel='$\gamma$', ylabel='$\\alpha$')

# ### WHOOPING CRANE DATASET CASE STUDY ###
t_data_w = [t for t in range(1938, 2010, 1)]
p_data_w = [9, 11, 13, 8, 10, 10, 9, 11, 13, 15, 15, 17, 15, 12, 10, 12, 11, 14, 12, 13, 16,
          17, 18, 19, 16, 16, 21, 22, 22, 24, 25, 28, 28, 30, 25, 25, 24, 29, 34, 36, 37,
          38, 39, 37, 36, 38, 43, 48, 55,  67, 69, 73, 73, 66, 68, 71, 66, 79, 80, 91,
          91, 94, 90, 88, 93, 97, 109, 110, 118, 133, 135, 132]


fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(t_data_b, p_data_b, 'k+')
axs.set(xlabel='Year')
axs.set(ylabel='Population')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('Black Robins')
tikzplotlib.save("robin_data.tex")


fig, axs = plt.subplots(1,1, figsize=(5,3))
axs.plot(t_data_w, p_data_w, 'k+')
axs.set(xlabel='Year')
axs.set(ylabel='Population')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title('Whooping Cranes')
tikzplotlib.save("crane_data.tex")

###############
est_v1 = bd.estimate(t_data_w, p_data_w, [2, 2, 0.05], [[0,10], [0,10], [0, 1]],
                    model='Verhulst',  known_p=[0], idx_known_p=[3], se_type='asymptotic',
                    opt_method='differential-evolution', seed=2021)
(est_v1.p).insert(0, 'Verhulst 1')
(est_v1.se).insert(0, 'Verhulst 1')
extra_v1 = ['Verhulst 1', est_v1.capacity[1], np.exp(est_v1.val)]
###############
est_v2 = bd.estimate(t_data_w, p_data_w, [2, 2, 0.05], [[0,10], [0,10], [0, 1]],
                    model='Verhulst',  known_p=[0], idx_known_p=[2], se_type='asymptotic',
                    opt_method='differential-evolution', seed=2021)
(est_v2.p).insert(0, 'Verhulst 2')
(est_v2.se).insert(0, 'Verhulst 2')
extra_v2 = ['Verhulst 2', est_v2.capacity[1], np.exp(est_v2.val)]
###############
est_r = bd.estimate(t_data_w, p_data_w, [2, 2, 0.05], [[0,10], [0,10], [0,1]], 
                    model='Ricker', known_p=[1], idx_known_p=[3], se_type='asymptotic', 
                    opt_method='differential-evolution', seed=2021)
(est_r.p).insert(0, 'Ricker')
(est_r.se).insert(0, 'Ricker')
extra_r = ['Ricker', est_r.capacity[1], np.exp(est_r.val)]
###############
est_b = bd.estimate(t_data_w, p_data_w, [2, 2, 2], [[0,10], [0,10], [0, 1]],
                      model='Hassell', known_p=[1], idx_known_p=[3], 
                      se_type='asymptotic', opt_method='differential-evolution',
                      seed=2021)
(est_b.p).insert(0, 'Beverton-Holt')
(est_b.se).insert(0, 'Beverton-Holt')
extra_b = ['Beverton-Holt', est_b.capacity[1], np.exp(est_b.val)]
###############
est_h = bd.estimate(t_data_w, p_data_w, [2, 2, 2], [[0,10], [0,10], [0, 1]],
                      model='Hassell', known_p=[2], idx_known_p=[3], 
                      se_type='asymptotic', opt_method='differential-evolution',
                      seed=2021)
(est_h.p).insert(0, 'Hassell')
(est_h.se).insert(0, 'Hassell')
extra_h = ['Hassell', est_h.capacity[1], np.exp(est_h.val)]
###############
est_mss = bd.estimate(t_data_w, p_data_w, [2, 2, 2], [[0,10], [0,10], [0, 1]], 
                      model='MS-S', known_p=[2], idx_known_p=[3], 
                      se_type='asymptotic', opt_method='differential-evolution', 
                      seed=2021)
(est_mss.p).insert(0, 'MS-S')
(est_mss.se).insert(0, 'MS-S')
extra_mss = ['MS-S', est_mss.capacity[1], np.exp(est_mss.val)]
###############
est_l = bd.estimate(t_data_w, p_data_w, [2]*2, [[0,10]]*2, model='linear', 
                    se_type='asymptotic', opt_method='differential-evolution', 
                    seed=2021)
(est_l.p).insert(0, 'linear')
(est_l.se).insert(0, 'linear')
extra_l = ['linear', 'n/a', np.exp(est_l.val)]
###############
est_lm = bd.estimate(t_data_w, p_data_w, [2]*3, [[0,10]]*3, model='linear-migration', 
                      se_type='asymptotic', opt_method='differential-evolution', 
                      seed=2021)
(est_lm.p).insert(0, 'linear-migration')
(est_lm.se).insert(0, 'linear-migration')
extra_lm = ['linear-migration', 'n/a', np.exp(est_lm.val)]
###############
###############
###############
print("\n Table: WHOOPING CRANE PARAMETER ESTIMATES")
print(tabulate([["Model", "gamma", "nu", "alpha/beta"],
            est_v1.p, est_v2.p, est_r.p, est_b.p, est_h.p, est_mss.p, est_l.p, est_lm.p],
          headers="firstrow", floatfmt=".4f", tablefmt='latex'))
###############
print("Table: WHOOPING CRANE STANDARD ERRORS")
print(tabulate([["Model", "gamma", "nu", "alpha", "beta/c"],
          est_v1.se, est_v2.se, est_r.se, est_b.se, est_h.se, est_mss.se, est_l.se, est_lm.se],
          headers="firstrow", floatfmt=".4f", tablefmt='latex'))
###############
print("Table: WHOOPING CRANE CARRYING CAPACITY AND LIKELIHOOD")
print(tabulate([["Model", "Capacity", "Likelihood"],
          extra_v1, extra_v2, extra_r, extra_b, extra_h, extra_mss, extra_l, extra_lm],
          headers="firstrow", tablefmt='latex'))
##############

bd.estimate(t_data_w, p_data_w, [0.1812, 0.1489], [[0,1], [0, 1]],
            model='linear-migration', known_p=[0.3157], idx_known_p=[2], 
            se_type='asymptotic', ci_plot=True, seed=2021, export='lm_cranes_ci_asy',
            xlabel='$\gamma$', ylabel='$\\nu$')

bd.estimate(t_data_w, p_data_w, [0.1812, 0.1489], [[0,1], [0, 1]],
            model='linear-migration', known_p=[0.3157], idx_known_p=[2], 
            se_type='simulated', ci_plot=True, seed=2021, export='lm_cranes_ci_sim',
            xlabel='$\gamma$', ylabel='$\\nu$')

bd.forecast('linear-migration', p_data_w[-1], np.arange(2009, 2051, 1), est_lm.p[1:], 
            cov=est_lm.cov, p_bounds=[[0,10]]*3, export="lm_cranes")

bd.forecast('MS-S', p_data_w[-1], np.arange(2009, 2051, 1), est_mss.p[1:], 
            cov=est_mss.cov, p_bounds=[[0,10]]*3, known_p=[2], idx_known_p=[3], export="mss_cranes")
