# SMC
 SMC implementation

This reposetory you can find an implemntation for the SMC.
All of the paper results can be repredused using this repo.

Import the two skripts to your enviormnt.
difine the following variabels:

expNum- number of experements to run.
dstDir- destination directory to save the results
path- working directory (must containr the distribution.txt file
distrebutions- list of distribution to run on: ['Uniform','Skewed','PowerLaw']
weights- link wieghts to run on:['uniform','power','linear']

AlgVS(weight,expNum)-preduses the results for SMC vs Other Strategies
MutliJobFixCapAlgVS(weight,expNum,cap)-preduses the results Multiple Workloads with fixed switch capacity, where cap is the switch capacity value.
MultiJobsMultiCap(weight,expNum,caps)-preduses the results Multiple Workloads with muliple switch capacity, where caps is the switch capacity list.
