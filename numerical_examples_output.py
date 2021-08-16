"""
This file contains the output as displayed by running 
'numerical_examples_code.py' with gpu_installed = True 
and complete_version = True. 
"""

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
