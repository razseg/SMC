#SMC implementation

In this repository, you can find an implementation for the SMC.
All of the paper results can be reproduced using this repo.

Import the two scripts to your environment.
define the following variables:

expNum- number of experiments to run.
dstDir- destination directory to save the results
path- working directory (must containr the distribution.txt file
distrebutions- list of distribution to run on: ['Uniform','Skewed','PowerLaw']
weights- link wieghts to run on:['uniform','power','linear']

AlgVS(weight,expNum)-produces the results for SMC vs. Other Strategies
MutliJobFixCapAlgVS(weight,expNum, cap)-produces the results Multiple Workloads with fixed switch capacity, where cap is the switch capacity value.
MultiJobsMultiCap(weight,expNum, caps)-produces the results Multiple Workloads with multiple switch capacity, where caps is the switch capacity list.
