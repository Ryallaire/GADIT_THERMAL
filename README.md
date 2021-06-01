# GADIT THERMAL
GPU Alternating Direction Implicit Thin film solver and Heat Solver

## About 
This code solves 3 pdes (i) thin-film equation, (ii) heat equation for the film, and (iii) heat equation for the substrate. All of (i), (ii) and (iii) are from https://arxiv.org/abs/2009.06536 (see model (A)). 

## Operating System Support
This version of GADIT Heat was developed and tested on the Ubuntu 18.04 platform and CUDA 10.2.89 and GCC 7.5.0. It has been adapted to work with HPC clusters (i) Stheno (NJIT) and (ii) Summit (ORNL). See below for compiling specifics.

## Compiling
Locally: command: 
	nvcc -lstdc++fs main.cu cuPentBatch.cu -lcusparse -lcublas |& grep error

Stheno: 
	module load cuda 
Summit:
	module load cuda/10.1.243
	module load gcc/6.4.0
	nvcc -lstdc++fs main.cu cuPentBatch.cu -lcusparse -lcublas |& grep error

## Minimal Setup

To implement GADIT, it is instructive to read the main.cu file for a minimal example of how to initialize and execute GADIT. To implement your own thin film model, you may follow the instructions in the model_default.h file and edit appropriately.  Note that the main.cu file should also be modified to match the new parameters. Some parameters are in defined in paramaters.h. To use the code in linux, many c++ structures needed to be placed into this file (linux didn't like templating very much). 

GADIT also backs up the solution at a user specified time interval; therefore,  simulations may be stopped and started without any significant loss. Note that this backup includes the following
(i) BackupSolutionData.bin
	This is the film thickness h(x,y,t) at backup time (default is every 5 minutes).
(ii) BackupFilmTemp.bin
	This is the film temperature T_f(x,y,t) at backup time.
(iii) BackupSubTemp.bin
	This is the substrate temperature T_s(x,y,z,t) at backup time. Note that this file should be the largest (n x m x p).
(iv) Backupinfo.bin
	This file contains all the timestep information necessary to continue a simulation.

### Input 
The only possible external input to GADIT is initial condition data. The data must be in binary 64-bit float format and place in input/InitialCondition.bin
of the executing directory, or the directory specified by the user. In addition (for now), the data must be padded by two ghost points on  each boundary e.g.  a n by m matrix becomes n+4 by m+4 matrix. There is no need to specify the ghost points.

### Output

Similar to the initial condition data,  the solution at the output times are in binary 64-bit float format and are padded with the ghost points. Solution data will be saved in the sub-folder 'output' of the executing directory or the directory specified by the user. 

In addition to the solution data, several files are created in the 'status' sub-directory.

#### Parameters.txt
A record of the parameters used to initialize GADIT.

#### Status.txt
Useful notifications about the status of GADIT. Primarily contains information of when the simulation was started and last backed up. 
The parameters data structure may be edit so that time step information is also outputted to the file. In addition, the data may also be outputted to the console.

#### Timesteps.bin
Binary 64-bit float data containing the timestep value at every point. 

#### NewtonIterationCounts.bin
Binary  64-bit integer data containing the amount of Newton iterations at each time step. 

#### GADITstatus.bin
Binary 32-bit integer data which records the failure states of Newton iteration scheme.  Also, records time step increases. See namespace newton_status in parameters.h for failure code designations.

#### GADITstatusIndices
Binary 64-bit integer data containing the timestep indices for records in GADITstatus.bin.

#### Temporary Files 
The sub-folder 'temp' contains a backup of solutions. There are two files, BackupSolutionData.bin, the solution data; and BackupSolutionData.bin, a copy of the timestep_manager object.


### Simulation Specific Information
Here you can include information specific to this simulation for record keeping.
Domain:
- Thick Substrate Simulation, Hs=10 (100nm), Nzs=p=15.
- Large domain: n=m=181*8

Intiial Condition:
- Read from file (created in matlab)

Timestepping:
- dt_init = 0.000001*paras.temporal.dt_out
- dt_out = 10.0
- t_end = 60*paras.temporal.dt_out (end at 600)
	--made 600 even though 500 should be plenty. Films dewet near 400, but this allows us to run the simulations longer if need be.

Dimensionless Parameters:
- CC = 1.0
- cK = 14.20277
- cD = 0.0
- Bi = 0.1

Material Parameters:
- tp = 18e-9		(pulse width)
- energy_0 = 1.4e3 	(laser fluence)

Switches:
- constant parameters = true		(constant material parameters)
- time_varying_surf_tens = false	(time varying surface tension)
- time_varying_visc = false		(time varying viscosity)
- spat_varying_visc = false		(spatially varying viscosity)
- nonlin_sub_heat = true		(Use Newton's method for TC varying; may want to turn this off for the simple case k=1)
- nonlin_grid = true			(nonuniform z-grid in substrate)
- sub_in_plane_diff = false 		(drop Txx and Tyy in substrate heat solver)
- TC_model = 1 (k=1)





