
// ----------------------------------------------------------------------------------
// Copyright 2016-2017 Michael-Angelo Yick-Hang Lam
//
// The development of this software was supported by the National Science Foundation
// (NSF) Grant Number DMS-1211713.
//
// This file is part of GADIT.
//
// GADIT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as published by
// the Free Software Foundation.
//
// GADIT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GADIT.  If not, see <http://www.gnu.org/licenses/>.
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
// Name:			main.cu
// Version: 		1.0
// Purpose:			Minimal example code of how to execute GADIT.
// ----------------------------------------------------------------------------------

#include "gadit_solver.h"

int main()
{

	// Allow switching between float and double precision.
	// Note: May remove in future versions and assume double precision.
	typedef double PRECISION;

	//select model ID and initial condition ID
	// see solver_template.h for list of values
	model::id const MODEL_ID = model::THERMAL1;
	initial_condition::id const IC_ID = initial_condition::LINEAR_WAVES;
	// initial_condition::id const IC_ID = initial_condition::LOAD_FROM_FILE;

	// select boundary conditions
	// NOTE: For now only symmetric boundary conditions are implemented 
	//       i.e. h_x=h_xxx=0. Will upload revision to the code in the following
	//       months that allow a cleaner implementation of multiple boundary condition.
	//       You may alter boundary_conditions.h and implement your own boundary conditions
	boundary_condtion_type const BC_X0 = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type const BC_Y0 = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type const BC_XN = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type const BC_YM = boundary_condtion_type::SYMMETRIC;

	// simplifying class reference
	typedef gadit_solver<PRECISION, MODEL_ID, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM> gadit_solver;

	// File contain all parameters that can be altered by the user. 
	parameters<PRECISION, MODEL_ID, IC_ID> paras;

	// Spatial partition parameters
	paras.spatial.ds = 0.099909159073125;//0.1;
	paras.spatial.x0 = 0.0;
	paras.spatial.n = 181*8;
	paras.spatial.y0 = 0.0;
	paras.spatial.m = 181*8;
	paras.spatial.p = 15;		//Number of spatial (z-direction) points in the substrate
	paras.spatial.Nzs = 15;		//Number of spatial (z-direction) points in the substrate (old)
	paras.spatial.th_sio2 = 10.0;
	paras.spatial.padding = 2;
	//paras.spatial.n_pad = paras.spatial.n + 2 * paras.spatial.padding;
	//paras.spatial.m_pad = paras.spatial.m + 2 * paras.spatial.padding;
	//paras.spatial.dzs = 0.01;


	

	// Model Parameters
	paras.model.cC = 1.0;									// Capillary Term in thin film equation (coefficient of grad(laplace(h)))
	paras.model.cN = 0.0; 									// Nematic Term in Thin film equation (previous value = 1.67)
	paras.model.cK = 14.20277;								// Disjoining Pressure term in Thin Film equation (coefficient of grad(h))
	paras.model.cD = 0.0;									// Marangoni term (coefficient of grad(gamma) or grad(T) )
	paras.model.b = 0.1;									// Precursor Thickness
	paras.model.beta = 1.0;									// Need to check this parameter
	paras.model.w = 0.05;									// Need to check this parameter
	paras.model.theta_c = 50.0;
	paras.model.vdw_n = 3.0;
	paras.model.vdw_m = 2.0;

	// Thermal Parameters
	paras.model.l_scl = 1.0e-8;								// Lateral Length Scale
	paras.model.h_scl = 1.0e-8;								// Vertical Length Scale
	paras.model.gamma_0 = 1.303;							// (N/m): Surface Tension at melting temperature
	paras.model.gamma_T = -1.0*0.23e-3;						// (N/m-K) dgamma/dT
	paras.model.rho_sio2 = 2.2e3;							// Substrate Density
	paras.model.rho_m = 8.0e3;								// Film Density
	paras.model.visc_0 = 0.0043;							// Base film viscosity (melting temperature)
	paras.model.Ceff_sio2 = 9.37e2;							// Substrate Heat Capacity (Pa s)
	paras.model.Ceff_m = 4.95e2;							// Film Heat Capacity (Pa s)
	paras.model.k_sio2 = 1.4;								// Substrate base thermal conductivity (room temperature)
	paras.model.k_m = 3.4e2;								// Film thermal conductivity
	paras.model.Temp_room = 300.0;							// Room temperature (K)
	paras.model.energy_0 = 1.4e3;							// Laser fluence (J/m^2)
	paras.model.alpha_m_inv = 11.09e-9;						// Film absorption length m^(-1)
	paras.model.alpha_r_inv = 12.0e-9;						// Film reflective length/ reflectivity parameter
	paras.model.reflect_coef = 0.3655;						// Reflectivity fitting parameter
	paras.model.zeta = 1.0;									// Scaling of Source
	paras.model.tp = 18.0e-9;								// Pulse width
	paras.model.E_act = 30500;								// Activation Energy for Viscosity
	paras.model.R = 8.3144598;								// Boltzmann constant
	paras.model.th_sio2 = 10.0;								// Dimensionless substrate thickness

	paras.model.alpha_s = 8.0e5;							// SiO_2 Heat transfer coefficient
	paras.model.alpha_E = 0.0;								// Heat transfer coefficient East (x=L, z>0)
	paras.model.alpha_W = 0.0;								// Heat transfer coefficient West (x=0, z>0)
	paras.model.alpha_N = 0.0;								// Heat transfer coefficient North (y=L, z>0)
	paras.model.alpha_south = 0.0;							// Heat transfer coefficient South (x=0, z>0)
	paras.model.alpha_SE = 0.0;								// Heat transfer coefficient Substrate, East (x=L,z<0)
	paras.model.alpha_SW = 0.0;								// Heat transfer coefficient Substrate, West (x=0, z<0) 	
	paras.model.alpha_SN = 0.0;								// Heat transfer coefficient Substrate, North (x=L,z<0)
	paras.model.alpha_SS = 0.0;								// Heat transfer coefficient Substrate, South(x=0, z<0) 	

	paras.model.surf_tens = paras.model.gamma_0;
	paras.model.visc = paras.model.visc_0;
	paras.model.temp_tol = 0.001;

	paras.model.film_evolution = false;						// Logical switch for turning on/off film evolution (initially off before melting)
	paras.model.melting_switch = true;						// Logical switch for waiting for film to melt before evolution moves
	paras.model.temp_adapt = false;							// Logical switch for temperature adaptivity (can have initially off)
	paras.model.output_sub_temp = true;						// Logical switch for outputting substrate temperature; WARNING: files may get big!

	// Logical Switches for Changing material parameters
	paras.model.constant_parameters = false;				// Logical switch for keeping material parameters constant or not
	paras.model.time_varying_surf_tens = true;				// Logical switch for time varying surface tension
	paras.model.time_varying_visc = true;					// Logical switch for time varying viscosity
	paras.model.spat_varying_visc = false;					// Logical switch for spatially varying viscosity (should not use both time-varying and spatially varying concurrentlly)

	// Logical switches for substrate solver
	paras.model.nonlin_sub_heat = true;						// Logical switch for including temperature varying thermal conductivity in substrate
	paras.model.nonlin_grid = true;							// Logical switch for using a nonlinear geometric z-grid in the substrate
	paras.model.sub_in_plane_diff = false;					// Logical switch for including Txx and Tyy in substrate heat solver

	paras.model.TC_model = 3;								// Choose Thermal conductivity model (1 = constant TC, 2 = cubic polynomial, 3 = cubic w/ sigmoids/smoothing)
	paras.model.T_eps_newt = 1e-9;							// Newton convergence tolerance for nonlinear substrate heat solver
	paras.model.n_iter_max = 10;							// Maximum number of iterations for Newton method convergence for substrate heat solver

	// paras.model.compute_derived_parameters();


	// Parameters for initial condition LINEAR_WAVES
	paras.initial.h0 = 1.0;									// Base film thickness
	paras.initial.epx = 0.01;								// x-perturbation magnitude
	paras.initial.nx = 2 * 8;								// nx/2 = # of waves in domain in x-direction
	paras.initial.epy = 0.01;								// y-perturbation magnitude
	paras.initial.ny = 2 * 8;								// ny/2 = # of waves in domain in y-direction

	// Temporal parameters
	paras.temporal.t_start = 0.0;							// Simulation start time
	paras.temporal.dt_out = 10.0;							// File outputting time (smaller = bigger file sizes)
	paras.temporal.t_end = paras.temporal.dt_out*80.0;		// Simulation end time


	// backup time for solution in minutes 
	paras.backup.updateTime = 5;

	// Add '/' to end if not using execution directory as root e.g. some_folder/
	paras.io.root_directory = "";

	// Toggle to control output of status of GADIT solver
	paras.io.is_console_output = true;
	paras.io.is_full_text_output = false;


	// It is not necessary the change the remaining parameters,
	// but feel free to do so.
	paras.newton.error_tolerence = pow(10,-10);//pow(10,-9);//pow(10,-9);//pow(10, -10);//pow(10, -10);

	// Testing shows 10 produces best effective time step
	// i.e. dt/interation_count
	paras.newton.max_iterations = 11;
	// Applies a minimum amount iterations with out convergence checks
	paras.newton.min_iterations = 0;

	paras.temporal.dt_min = pow(10, -13);
	paras.temporal.dt_max = 0.01;

	// set large to prevent excessive dt increase
	// that with results in immediate newton convergence
	// failure within thresholds.
	paras.temporal.min_stable_step = 10;//500;//10;//500;//500;
	// dt is allowed to increase exponentially once min_step is
	// reach. After failure,  min_stable_steps much be achieved
	// before dt can be increased again.
	paras.temporal.dt_ratio_increase = 1.01;//1.07;//1.02;//1.07;
	paras.temporal.dt_ratio_decrease = 2.0;//1.05;//2.0;//1.05;

	// setting to a very small will only affect the
	// start up of GADIT. GADIT allows exponential growth
	// of the time step
	// paras.temporal.dt_init = 0.00001;
	paras.temporal.dt_thermal = 0.0000000001;					// Minimum time increment to solve thermal problem
	if (!paras.model.film_evolution) {
		paras.temporal.dt_init = 0.1; //paras.temporal.dt_thermal;
	}
	paras.temporal.dt_init = 0.01;								// Initial time step

	// initializes solver and evolve solution
	gadit_solver *solver;
	solver = new gadit_solver();

	solver->initialize(paras);
	solver->solve_model_evolution();

	return 0;
}

