# Descriptions

## get_best_hardware.py

Given a database, an FPGA device, a resource constraint, and a frequency setting,
    search the parameter space (nlist, nprobe, OPQ_enable) and find the best
    hardware solution.

Constraint:

* The maximum usable HBM number for stage 2 and 5 are 20 in total for U280/U50
  * experiments show that more than 20 banks can lead to routing errors
* The maximum stage 5 PE number is limited
  * for K=10,100, the limitation is 36 (39 can lead to routing errors according to experiments)

## get_hardware_performance.py

Input a set of hardware settings (config.yaml), predict the performance and resource 
    consumption by the performance  model.
