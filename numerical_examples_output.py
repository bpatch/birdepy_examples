"""
This file contains example output as displayed by running 
'numerical_examples_code.py' with gpu_installed = True 
and complete_version = True. 
"""

#     Table: Simulation CPU times
# \begin{tabular}{lrrrrr}
# \hline
#                      &   exact &     ea &     ma &     gwa &    gpu \\
# \hline
#  Compute time (secs) & 73.7610 & 3.8110 & 7.1890 & 13.7780 & 0.1090 \\
# \hline
# \end{tabular}
# C:\Users\brend\anaconda3\envs\test7\lib\site-packages\birdepy\probability_gwasa.py:338: RuntimeWarning: Probability not in [0, 1] computed, some output has been replaced by a default value.  Results may be unreliable.
#   warnings.warn("Probability not in [0, 1] computed, "
# Table: Small population Verhulst CPU times
# \begin{tabular}{lrrrrrrrrr}
# \hline
#                      &    sim &   expm &   uniform &   Erlang &    ilt &     da &    oua &    gwa &   gwasa \\
# \hline
#  Compute time (secs) & 0.5840 & 0.0070 &    0.7700 &   0.0030 & 2.0500 & 0.0050 & 0.0040 & 0.0250 &  0.0030 \\
# \hline
# \end{tabular}
# Basic ABC estimate is [0.7468397861173611], with standard error [0.1033682469591971]computed in 74.70699763298035 seconds.
# Dynamic ABC estimate is [0.7620827322497084], with standard error [0.05331307382209087]computed in 1853.4449698925018 seconds.
# C:\Users\brend\anaconda3\envs\test7\lib\site-packages\scipy\optimize\optimize.py:282: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
#   warnings.warn("Values in x were outside bounds during a "
# C:\Users\brend\anaconda3\envs\test7\lib\site-packages\birdepy\interface_estimate.py:530: RuntimeWarning: invalid value encountered in sqrt
#   se = list(np.sqrt(np.diag(cov)))
# Table: DNM Estimates
# \begin{tabular}{lrrr}
# \hline
#          &   gamma &     nu &   alpha \\
# \hline
#  da      &  0.7817 & 0.3906 &  0.0252 \\
#  Erlang  &  0.7824 & 0.3873 &  0.0250 \\
#  expm    &  0.7810 & 0.3867 &  0.0250 \\
#  gwa     &  0.4111 & 0.3414 &  0.0076 \\
#  gwasa   &  0.4198 & 0.3494 &  0.0075 \\
#  ilt     &  0.7782 & 0.3852 &  0.0251 \\
#  oua     &  0.6528 & 0.3716 &  0.0229 \\
#  uniform &  0.7811 & 0.3867 &  0.0250 \\
# \hline
# \end{tabular}
# Table: DNM Standard Errors
# \begin{tabular}{lrrr}
# \hline
#          &   gamma &     nu &    alpha \\
# \hline
#  da      &  0.0612 & 0.0292 &   0.0013 \\
#  Erlang  &  0.0653 & 0.0294 &   0.0013 \\
#  expm    &  0.0650 & 0.0292 &   0.0013 \\
#  gwa     &  0.0432 & 0.0228 &   0.0041 \\
#  gwasa   &  0.0438 & 0.0239 &   0.0040 \\
#  ilt     &  0.0047 & 0.0052 & nan      \\
#  oua     &  0.0570 & 0.0298 &   0.0016 \\
#  uniform &  0.0650 & 0.0292 &   0.0013 \\
# \hline
# \end{tabular}
# Table: DNM Compute Times
# \begin{tabular}{lr}
# \hline
#  Method   &   Compute time (secs) \\
# \hline
#  da       &               89.8750 \\
#  Erlang   &                0.8610 \\
#  expm     &                1.0960 \\
#  gwa      &               88.9520 \\
#  gwasa    &                7.3920 \\
#  ilt      &             4738.9250 \\
#  oua      &               15.7860 \\
#  uniform  &                2.8480 \\
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
#  expm & cg       &                4.6590 \\
#  expm & none     &                4.3630 \\
#  expm & Lange    &                4.4360 \\
#  expm & qn1      &               11.0230 \\
#  expm & qn2      &                3.5980 \\
#  ilt  & cg       &             9409.6620 \\
#  ilt  & none     &             8338.6820 \\
#  ilt  & Lange    &             9518.8610 \\
#  ilt  & qn1      &            19994.9424 \\
#  ilt  & qn2      &             6198.4972 \\
#  num  & cg       &               12.3250 \\
#  num  & none     &               87.9390 \\
#  num  & Lange    &               27.1730 \\
#  num  & qn1      &               46.6500 \\
#  num  & qn2      &                9.8320 \\
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
#  expm &  0.0383 & 0.0241 &  0.0021 \\
#  fm   &  0.0367 & 0.0257 &  0.0025 \\
#  gwa  &  0.0480 & 0.0354 &  0.0024 \\
# \hline
# \end{tabular}
# Table: LSE Compute Times
# \begin{tabular}{lr}
# \hline
#  Method   &   Compute time (secs) \\
# \hline
#  expm     &               21.4170 \\
#  fm       &              484.6330 \\
#  gwa      &               15.9930 \\
# \hline
# \end{tabular}
# Overall script compute time: 61270.87859225273
