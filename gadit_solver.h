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
// Name:		gadit_solver.h
// Version: 	1.0
// Purpose:		Main interface to solver. 
// Output:		Output solutions with ghost points.
// ----------------------------------------------------------------------------------

#ifndef GADIT_SOLVER
#define GADIT_SOLVER


#include "cuda_runtime.h"

#include "initial_condition_list.h"
#include "newton_iterative_method.h"
#include "backup_manager.h"
#include "parameters.h"
#include "timestep_manager.h"
#include "status_logger.h"


#include "output.h"
#include "subf_temp.h"

#include <chrono>
#include <ctime>
#include <algorithm>
#include <math.h>
#include <cmath>


__global__ void find_minimum_kernel(double *array, double *min, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];

	 *min = 1000.0;
	// double temp = 100000.0;
	double temp = 100000.0;
	while(index + offset < n){
		temp = fmin(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmin(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*min = fmin((*min), cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

__global__ void update_surf_tens(double* G, double* res_temp, int n_pad, int padding, int n, int m, double gamma_ratio, double Temp_melt)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k_warp, k_temp;
	if(idx < n && idy < m)
	{
		k_warp = (n_pad)*idx + padding + 2 * n_pad + idy;
		k_temp = idx + n * idy;
		G[k_warp] = 1e0 + gamma_ratio*(res_temp[k_temp] - Temp_melt);
	}

}

__global__ void update_visc_rat(double* visc_rat, double* res_temp, int n_pad, int padding, int n, int m, double E_act, double R, double Temp_scl, double T_melt)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k_warp, k_temp;
	if(idx < n && idy < m)
	{
		k_warp = (n_pad)*idx + padding + 2 * n_pad + idy;
		k_temp = idx + n * idy;
		visc_rat[k_warp] = 1e0*exp(E_act/(R*Temp_scl)*(1.0/(res_temp[k_temp])-1.0/T_melt));
	}

}

__global__ void store_h_at_current_time_step(double* A, double* B, int n_pad, int m_pad)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k;
	if (idx < n_pad && idy < m_pad)
	{
		k = idx + n_pad * idy;						// Note this index should loop over each element of h, even padding
		B[k] = A[k];
	}
}

__global__ void reset_h(double* A, double* B, int n_pad, int m_pad)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k;
	if (idx < n_pad && idy < m_pad)
	{
		k = idx + n_pad * idy;						// Note this index should loop over each element of h, even padding
		B[k] = A[k];
	}
}

void compute_lateral_coef(double Biot, double &D, double &E, double Temp_amb)
{
	//== This function computes the coefficients for the ghost points e.g. T_0 = D T_1 + E
	D = (1.0-Biot)/(1.0+Biot);
	E = (2.0*Biot)/(1.0+Biot)*Temp_amb;
}

template <typename DATATYPE, model::id MODEL_ID , initial_condition::id IC_ID, 
	boundary_condtion_type BC_X0, boundary_condtion_type BC_XN,
	boundary_condtion_type BC_Y0, boundary_condtion_type BC_YM> class gadit_solver
{
public:

	gadit_solver(){};

	void initialize(parameters<DATATYPE,MODEL_ID,IC_ID> paras )
	{
		paras.spatial.compute_derived_parameters();										//Initialize Spatial Parameters
		paras.model.compute_derived_parameters();										//Initialize Model Parameters
		this->paras = paras;

		int padding = cuda_parameters::CELL_BORDER_PADDING;								//Set Cell_border Padding size

		dims.set_dimensions(paras.spatial.n, paras.spatial.m, paras.spatial.p, padding);					//Set Spatial dimensions and padding

		int  initial_step_size = nonlinear_penta_solver::INITIAL_STEP_SIZE;				//Set Initial timestep size
		int  down_solve_sub_loop_size = nonlinear_penta_solver::DOWN_SOLVE_SUB_LOOP;	
		int  thread_size = cuda_parameters::PENTA_LU_LINE_THREAD_SIZE;					//Thread Size for GPU

		dims.set_penta_dimension(initial_step_size, down_solve_sub_loop_size, thread_size);		//Set Size of Matrices in Penta Solver

		int reduction_block_size = cuda_parameters::SIMPLE_SQUARE_BLOCK;				//Set reduced block size for GPU

		dims.set_reduction_dimension(reduction_block_size);

		int Jy_F_y_subloop_size = cuda_parameters::SOLVE_JY_SUBLOOP_SIZE;
		int Jy_F_y_thread_size = cuda_parameters::SOLVE_JY_THREAD_SIZE;

		dims.set_Jx_F_dimension(Jy_F_y_thread_size, Jy_F_y_subloop_size);

		int simple_block_size = cuda_parameters::SIMPLE_SQUARE_BLOCK;

		dims.set_simple_block_dimension(simple_block_size);

		u_ws.initalize_memory(dims);	

		ker_launch_paras.initalize(dims);

		for (int i = 0; i < dims.n; i++)
		{
			u_ws.x->data_host[i] = paras.spatial.x0 + (double(i) + 0.5)*paras.spatial.ds;
		}
		for (int j = 0; j < dims.m; j++)
		{
			u_ws.y->data_host[j] = paras.spatial.y0 + (double(j) + 0.5)*paras.spatial.ds;
		}
	}


	void solve_model_evolution()
	{

		string outputString;
		timestep_manager<DATATYPE> t_mang;
		backup_manager<DATATYPE> sol_b_mang;
		status_logger<DATATYPE> log_file;

		
		newton_status::status n_status;
		timestep_manager_status::status t_status;

		sol_b_mang.initialize( paras.backup.updateTime );

		bool isFirstRun = !fileExists( file_directories::backupFileInfo );

		// loading data if temporary data exists.
		if ( isFirstRun )
		{
			file_directories::make_directories(paras.io.root_directory);
			file_directories::clean();
			write_to_new_file( paras.io.root_directory + file_directories::parameterData, paras.to_string() , false);

			t_mang.initialize( paras.temporal );
			initial_condition_list::compute<DATATYPE,IC_ID>( u_ws.x , u_ws.y , u_ws.h , dims , paras.spatial , paras.initial , paras.io);

			outputString =  get_time_stamp() + "Started simulations from initial condition.";
			write_to_new_file(paras.io.root_directory + file_directories::statusData , outputString, paras.io.is_console_output);

		}
		else
		{		

			load_object<timestep_manager<DATATYPE>>( paras.io.root_directory + file_directories::backupFileInfo , t_mang );
			load_binary ( file_directories::backupSolution , u_ws.h->data_host , dims.n_pad , dims.m_pad );
			
			char buff[100];				
			sprintf(buff,"Continuing simulations from backup data at t = %11.10E." , t_mang.get_current_time() );
			outputString =  get_time_stamp() + buff;
			write_to_old_file( paras.io.root_directory + file_directories::statusData , outputString , paras.io.is_console_output );
		}

		memory_manager::copyHostToDevice<DATATYPE>(u_ws.h);

		double t_thermal_update = t_mang.get_current_time();

		//== Initialize the temperature profiles===============================//
		double *res_temp, *T_int, *Tsub;		// Define temperature pointers on cpu for initialization

		res_temp = new double [dims.m * dims.n];
		T_int = new double [dims.m * dims.n];
		Tsub = new double [dims.m * dims.n * dims.p];

		//== For finding maximum of an array
		double *d_max;
		int *d_mutex;
		double *h_max;
		h_max = (double*)malloc(sizeof(double));
		cudaMalloc((void**)&d_max, sizeof(double));
		cudaMalloc((void**)&d_mutex, sizeof(int));
		cudaMemset(d_max, 0, sizeof(double));
		cudaMemset(d_mutex, 0, sizeof(double));

		//== For finding minimum of an array
		double *d_min;
		int *d_mutex_min;
		double *h_min;
		h_min = (double*)malloc(sizeof(double));
		cudaMalloc((void**)&d_min, sizeof(double));
		cudaMalloc((void**)&d_mutex_min, sizeof(int));
		cudaMemset(d_min, 1000, sizeof(double));
		cudaMemset(d_mutex_min, 0, sizeof(double));




		//== Initialize non-uniform z-grid for substrate
		double dz_r = 1.5;//1.5;
		double dz_min;
		double dz_sum = 0;
		for(int j=0; j<=dims.p-2; j++)
		{
			dz_sum += pow(dz_r, j);
		}
		dz_sum += pow(dz_r, dims.p-1)*0.5;
		dz_min = paras.spatial.th_sio2 / dz_sum ; 

		//== Initialize dzs array
		double *dzs_var;
		//== We will primarily use p-1 spatial steps. Including last term here for boundary
		dzs_var = new double[dims.p];

		dzs_var[0] = dz_min;
		for(int j=1; j<dims.p; j++)
		{
			dzs_var[j] = dz_min*pow(dz_r, j);
		}

		double *dz_A, *dz_B, *dz_C, *dz_D;
		double *T_int_coef; 
		dz_A = new double [dims.p-1];
		dz_B = new double [dims.p-1];
		dz_C = new double [dims.p-1];
		dz_D = new double [dims.p-1];
		T_int_coef = new double[3];
		double dzs_sum;

		for(int j=0; j<dims.p-1;  j++)
		{
			dzs_sum = dzs_var[j] + dzs_var[j+1];
			dz_A[j] = 2.0 / ((dzs_sum)*dzs_var[j]);
			dz_B[j] = -2.0 / (dzs_var[j] * dzs_var[j+1]);
			dz_C[j] = 2.0 / ((dzs_sum)*dzs_var[j+1]);

			if(paras.model.nonlin_sub_heat){
				dz_A[j] *= paras.model.K2;
				dz_B[j] *= paras.model.K2;
				dz_C[j] *= paras.model.K2;
				dz_D[j] = 2.0*paras.model.K2 / (dzs_sum * dzs_sum);
			}
		}
		T_int_coef[0] = (2*dzs_var[0] + dzs_var[1]) / (dzs_var[0]*(dzs_var[0] + dzs_var[1]));
		T_int_coef[1] =  -(1.0 / dzs_var[0] + 1.0 / dzs_var[1]);
		T_int_coef[2] = (dzs_var[0]) / (dzs_var[1] * (dzs_var[0] + dzs_var[1]));


		//== Define Biot Numbers for lateral boundaries
		paras.model.Bi_E_star = 0.5*paras.model.Bi_E*paras.spatial.dx;
		paras.model.Bi_W_star = 0.5*paras.model.Bi_W*paras.spatial.dx;
		paras.model.Bi_N_star = 0.5*paras.model.Bi_N*paras.spatial.dy;
		paras.model.Bi_S_star = 0.5*paras.model.Bi_S*paras.spatial.dy;
		paras.model.Bi_SE_star = 0.5*paras.model.Bi_SE*paras.spatial.dx;
		paras.model.Bi_SW_star = 0.5*paras.model.Bi_SW*paras.spatial.dx;
		paras.model.Bi_SN_star = 0.5*paras.model.Bi_SN*paras.spatial.dy;
		paras.model.Bi_SS_star = 0.5*paras.model.Bi_SS*paras.spatial.dy;

		//== Define ghost point coefficients
		compute_lateral_coef(paras.model.Bi_E_star, paras.model.D_E, paras.model.E_E, paras.model.Temp_amb);
		compute_lateral_coef(paras.model.Bi_W_star, paras.model.D_W, paras.model.E_W, paras.model.Temp_amb);
		compute_lateral_coef(paras.model.Bi_N_star, paras.model.D_N, paras.model.E_N, paras.model.Temp_amb);
		compute_lateral_coef(paras.model.Bi_S_star, paras.model.D_S, paras.model.E_S, paras.model.Temp_amb);
		compute_lateral_coef(paras.model.Bi_SE_star, paras.model.D_SE, paras.model.E_SE, paras.model.Temp_amb);
		compute_lateral_coef(paras.model.Bi_SW_star, paras.model.D_SW, paras.model.E_SW, paras.model.Temp_amb);
		compute_lateral_coef(paras.model.Bi_SN_star, paras.model.D_SN, paras.model.E_SN, paras.model.Temp_amb);
		compute_lateral_coef(paras.model.Bi_SS_star, paras.model.D_SS, paras.model.E_SS, paras.model.Temp_amb);

		//== Initialize memory for heat code
		//== Define matrices as arrays on device
		double *dev_res_temp, *dev_res_temp_trans, *dev_res_temp_prev, *dev_res_temp_trans_prev;
		double *dev_Tsub, *dev_Tsub_prev, *dev_T_int, *dev_T_int_prev, *dev_T_int_corrector;
		double *dev_res_prev, *dev_res_nxt;
		double *dev_res_trans, *dev_res_nxt_trans;
		double *dev_courant;
		double *dev_err_local;
		double *dev_muzA, *dev_muzB, *dev_muzC, *dev_muzD;
		//============================================================

		//== Variables for Substrate ADI Method;
		double *dev_Tsub_int1, *dev_Tsub_int2;
		double *dev_Tsub_ikj, *dev_Tsub_jki;
		double *dev_Tsub_int1_jki, *dev_Tsub_int2_kij;
		double *Az_ijk_stored, *Az_kij_stored;
		double *Ay_ijk_stored, *Ay_jki_stored;
		//=================================================


		//== Define coefficients for T_int derivative on non-uniform grid
		double *dev_T_int_coef;

		//== Variables for matrix and vector //
		double *dev_matx_sub, *dev_matx_main, *dev_matx_sup, *dev_vecx;
		double *dev_maty_sub, *dev_maty_main, *dev_maty_sup, *dev_vecy;
		double *dev_vecx_stored, *dev_matx_sup_stored, *dev_maty_sup_stored;
		double *Qterm_stored;

		//== Average temperature variables for gpu reduction sum
		double* dev_partial_T_avg, *partial_T_avg;
		

		const int TPB = 256;
		const int BPG = min( 32, (dims.n*dims.m+TPB-1) / TPB );

		partial_T_avg = (double*)malloc( BPG*sizeof(double) );

		// allocate the memory on the GPU
		//== In practice many of these variables should not be declared unless used.
		cudaMalloc( (void**)&dev_partial_T_avg, BPG*sizeof(double) );

		//== Define adix matrix and vector of length Nx
		cudaMalloc ( (void**)&dev_matx_sub, (dims.n*dims.m)*sizeof(double));		//== Note this was changed to n on 5/23/20
		cudaMalloc ( (void**)&dev_matx_main, (dims.n*dims.m)*sizeof(double));
		cudaMalloc ( (void**)&dev_matx_sup, (dims.n*dims.m)*sizeof(double));
		cudaMalloc ( (void**)&dev_vecx, (dims.n*dims.m)*sizeof(double));
		cudaMalloc ( (void**)&dev_matx_sup_stored, (dims.n*dims.m)*sizeof(double));
		cudaMalloc ( (void**)&dev_vecx_stored, (dims.n*dims.m)*sizeof(double));
		cudaMalloc ( (void**)&dev_maty_sup_stored, (dims.n*dims.m)*sizeof(double));
		cudaMalloc ( (void**)&Qterm_stored, (dims.n*dims.m)*sizeof(double));

		//== Define adiy matrix and vector of length Ny
		cudaMalloc ( (void**)&dev_maty_sub, (dims.m*dims.n)*sizeof(double));
		cudaMalloc ( (void**)&dev_maty_main, (dims.m*dims.n)*sizeof(double));
		cudaMalloc ( (void**)&dev_maty_sup, (dims.m*dims.n)*sizeof(double));
		cudaMalloc ( (void**)&dev_vecy, (dims.m*dims.n)*sizeof(double));

		//== Allocate memory on device for main variables
		cudaMalloc( (void**)&dev_res_temp, (dims.n*dims.m)*sizeof(double) );			    // Define res_temp on device: size N*M
		cudaMalloc( (void**)&dev_res_temp_trans, (dims.n*dims.m)*sizeof(double) );	    	// Define res_temp transposed on device: size N*M
		cudaMalloc( (void**)&dev_res_temp_prev, (dims.n*dims.m)*sizeof(double) );			// Define res_temp at time t_{n} on device: size N*M
		cudaMalloc( (void**)&dev_res_temp_trans_prev, (dims.n*dims.m)*sizeof(double) );		// Define res_temp tranposed at time t_{n} on device: size N*M
		cudaMalloc( (void**)&dev_Tsub, (dims.n*dims.m*dims.p)*sizeof(double) );			    // Define T_sub on device: size N*M*P
		cudaMalloc( (void**)&dev_Tsub_prev, (dims.n*dims.m*dims.p)*sizeof(double) );		// Define T_sub on device at time t_{n}: size N*M*P
		cudaMalloc( (void**)&dev_T_int, (dims.n*dims.m)*sizeof(double) );				    // Define T_int on device: size N*M
		cudaMalloc( (void**)&dev_T_int_prev, (dims.n*dims.m)*sizeof(double) );				// Define T_int on device: size N*M
		cudaMalloc( (void**)&dev_res_prev, (dims.n_pad*dims.m_pad)*sizeof(double) );		// Define res on device; size N_pad*M_pad
		cudaMalloc ( (void**)&dev_courant, (5)*sizeof(double));								// Define courant numbers on device; size 5
		cudaMalloc ( (void**)&dev_res_trans, (dims.n_pad*dims.m_pad)*sizeof(double));		// Define h transposed on device; size N_pad*M_pad
		cudaMalloc ( (void**)&dev_res_nxt_trans, (dims.n_pad*dims.m_pad)*sizeof(double));	// Define h tranposed at time t_{n+1} on device; size N_pad*M_pad
		cudaMalloc( (void**)&dev_err_local, (dims.n*dims.m)*sizeof(double) );			    // Define local error on device: size N*M
		cudaMalloc( (void**)&dev_T_int_corrector, (dims.n*dims.m)*sizeof(double) );			// Define T_int at correction stage	

		cudaMalloc ( (void**)&dev_muzA, (dims.p-1)*sizeof(double));
		cudaMalloc ( (void**)&dev_muzB, (dims.p-1)*sizeof(double));
		cudaMalloc ( (void**)&dev_muzC, (dims.p-1)*sizeof(double));
		cudaMalloc ( (void**)&dev_muzD, (dims.p-1)*sizeof(double));

		cudaMalloc ( (void**)&dev_T_int_coef, (3)*sizeof(double));

		cudaMalloc( (void**)&dev_Tsub_int1, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&dev_Tsub_int2, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&dev_Tsub_ikj, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&dev_Tsub_jki, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&dev_Tsub_int1_jki, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&dev_Tsub_int2_kij, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&Az_ijk_stored, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&Az_kij_stored, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&Ay_ijk_stored, (dims.n*dims.m*(dims.p-1))*sizeof(double) );
		cudaMalloc( (void**)&Ay_jki_stored, (dims.n*dims.m*(dims.p-1))*sizeof(double) );		
		// cudaMalloc( (void**)&dev_Tzz_stored, (dims.n*dims.m*dims.p)*sizeof(double) );
		// cudaMalloc( (void**)&dev_Tyy_stored, (dims.n*dims.m*dims.p)*sizeof(double) );


		double min_temp;
		int melt_iter=1;
		int solid_iter=1;
		int k_warp;
		int temp_dbg = 1;					// Initialize temperature debugger to 0. Does not initially check for error. This can be changed.
		int temp_status = 0;
		int k_temp, sub_id;
		dim3 tPB_h(min(dims.n_pad,32),min(dims.m_pad,32));
    	dim3 nB_h((dims.n_pad+tPB_h.x-1)/tPB_h.x, (dims.m_pad+tPB_h.y-1)/tPB_h.y);

		//== For update of surface tension , G=gamma
		dim3 tpb_G(min(dims.n,32),min(dims.m,32));
		dim3 num_blocks_G((dims.n+tpb_G.x-1)/tpb_G.x, (dims.m+tpb_G.y-1)/tpb_G.y );
		
		//== Run only if it is the first run
		if( isFirstRun ){
		//== Initialize starting values of temperature variables
		for (int i=0; i<dims.n; i++)
		{			
			for (int j=0; j<dims.m; j++)
			{
				k_temp = i + dims.n * j;
				k_warp = (dims.n_pad)*i + dims.padding + 2 * dims.n_pad + j;
				k_warp = i + dims.padding + dims.n_pad*(dims.padding + j);

				res_temp[k_temp] = paras.model.Temp_room/paras.model.T_scl;
				u_ws.gamma->data_host[k_warp] = 1.0 + paras.model.gamma_ratio*(res_temp[k_temp] - paras.model.Temp_melt);
				u_ws.visc_rat->data_host[k_warp] = 1.0;			//==Initialize the viscosity ratio to 1

				for (int k=0; k < dims.p; k++)
				{
					sub_id = k + dims.p * i + dims.n * dims.p * j;
					Tsub[sub_id] = paras.model.Temp_room/paras.model.T_scl;				//== Initialize Substrate Temperature Here to Room Temperature
				}

				sub_id = dims.p * i + dims.n * dims.p * j; // k=0
				// T_int[k_temp] = (3 * Tsub[sub_id] - 4 * Tsub[sub_id + 1] + Tsub[sub_id + 2]) * (paras.spatial.inv_dzst2);
				T_int[k_temp] = T_int_coef[0] * Tsub[sub_id] + T_int_coef[1] * Tsub[sub_id + 1] + T_int_coef[2] * Tsub[sub_id + 2];
			}
		}

		//====================================================================//
		} else{

			//== Read in film and substrate temperatures
			load_binary ( file_directories::backupfilmtemp , res_temp , dims.n , dims.m );
			load_binary ( file_directories::backupsubtemp , Tsub , dims.n*dims.m , dims.p );

			//== Recompute surface tension, gamma at current temperature, res_temp
			for (int i=0; i<dims.n; i++)
			{			
				for (int j=0; j<dims.m; j++)
				{
					k_temp = i + dims.n * j;
					k_warp = (dims.n_pad)*i + dims.padding + 2 * dims.n_pad + j;
					u_ws.gamma->data_host[k_warp] = 1.0 + paras.model.gamma_ratio*(res_temp[k_temp] - paras.model.Temp_melt);
					T_int[k_temp] = T_int_coef[0] * Tsub[sub_id] + T_int_coef[1] * Tsub[sub_id + 1] + T_int_coef[2] * Tsub[sub_id + 2];
					u_ws.visc_rat->data_host[k_warp] = 1.0*exp(paras.model.E_act/(paras.model.R*paras.model.T_scl)*(1.0/(res_temp[k_temp])-1.0/paras.model.Temp_melt));
				}
			}

			//== Manually set film and temp adaptivity
			paras.model.temp_adapt = true;
			paras.model.temp_adapt = false;
			paras.model.film_evolution = true;

			//== Update coefficients based on current temperature
			//== First compute average temperature
			double T_avg = 0.0;
			for (int j=0; j < dims.m; j++)
			{
				for (int i=0; i < dims.n; i++)
				{
					T_avg += res_temp[i+dims.n*j];
				}
			}
			T_avg /= (dims.n * dims.m);
			change_mat_par(T_avg, paras.model);

		}

		//== Copy gamma from host to device for initialization
		memory_manager::copyHostToDevice<DATATYPE>(u_ws.gamma);
		memory_manager::copyHostToDevice<DATATYPE>(u_ws.visc_rat);
		
		//== Copy Memory from host pointers to device pointers
		cudaMemcpy(dev_res_temp, res_temp, (dims.n*dims.m)*sizeof(double), cudaMemcpyHostToDevice);						// Store film temperature on device
		cudaMemcpy(dev_Tsub, Tsub, (dims.n*dims.m*dims.p)*sizeof(double), cudaMemcpyHostToDevice);						// Store Tsub on device
		cudaMemcpy(dev_T_int, T_int, (dims.n*dims.m)*sizeof(double), cudaMemcpyHostToDevice);							// Store T_int on device
		cudaMemcpy(dev_res_prev, u_ws.h->data_host, (dims.n_pad*dims.m_pad)*sizeof(double), cudaMemcpyHostToDevice);	// Store h at previous value for heat solver
		cudaMemcpy(dev_T_int_corrector, T_int, (dims.n*dims.m)*sizeof(double), cudaMemcpyHostToDevice);					// Initialize T_int_corrector on device
		cudaMemcpy(dev_T_int_coef, T_int_coef, (3)*sizeof(double), cudaMemcpyHostToDevice);								// Transfer coefficients for T_int derivative

		char buff2[100]; 
		std::string outputFileDir2;


		//===============================================================================//
		//== Print the following at t=0 =================================================//
		//===============================================================================//

		//== Print film thickness on host of size n_pad*m_pad
		sprintf(buff2, "/solution_%07d.bin", t_mang.get_next_output_index());
		outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
		memory_manager::copyDeviceToHost<DATATYPE>(u_ws.h);
		output_binary(outputFileDir2, u_ws.h->data_host, dims.n_pad, dims.m_pad);

		//== Print Surface tension gamma of size n_pad*m_pad
		// sprintf(buff2, "/gamma_%07d.bin", t_mang.get_next_output_index());
		// outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
		// memory_manager::copyDeviceToHost<DATATYPE>(u_ws.gamma);
		// output_binary(outputFileDir2, u_ws.gamma->data_host, dims.n_pad, dims.m_pad);

		//== Print film temperature of size n*m
		sprintf(buff2, "/filmtemp_%07d.bin", t_mang.get_next_output_index());
		outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
		output_binary(outputFileDir2, res_temp, dims.n, dims.m);

		//== Print T_int as a vector of length n*m
		// sprintf(buff2, "/T_int_%07d.bin", t_mang.get_next_output_index());
		// outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
		// output_binary(outputFileDir2, T_int, dims.n, dims.m);		
		//output_binary2(outputFileDir2, T_int, dims.n, dims.m);

		//== Print Tsub as a vector of length n*m*p
		if(paras.model.output_sub_temp){
			sprintf(buff2, "/Tsub_%07d.bin", t_mang.get_next_output_index());
			outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
			output_binary(outputFileDir2, Tsub , dims.n*dims.m*dims.p , 1);
		}


		char buff5[100];
		t_status = timestep_manager_status::SUCCESS;

		int adapt_starter = 1;

		//== Begin time loop
		while( t_mang.is_not_completed() ) 
		{	
			// cudaEventRecord(start);
			size_t newton_count;
			//== Initially set temp_dbg = 1
			//== If heat solved successfully temp_dbg == 0 and this while loop will end
			temp_dbg = 1;

			//== Repeat while loop until heat is successfully solved
			while (temp_dbg == 1)
			{

				//== Store h at current time before updating
				store_h_at_current_time_step<<<nB_h, tPB_h>>>(u_ws.h->data_device, dev_res_prev, dims.n_pad, dims.m_pad);

				//== Check if we should wait for the film to melt.
					//-- melting_switch = true => wait for melting to evolve film
					//---- if T_avg > T_melt, then film_evolution = true
					//-- melting switch = false => allow film to evolve immediately
				if (paras.model.film_evolution)
				{
	
					//== Update Film thickness at time t^{n+1}
					newton_iterative_method::solve_time_step<DATATYPE, MODEL_ID, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM>(u_ws, dims, paras, ker_launch_paras, t_mang.get_timestep(), newton_count, n_status);

				}
				else {

					//== If film evolution is turned off, automatically set newton_status to "FROZEN" so time stepper proceeds
					n_status = newton_status::FROZEN;

				}

				//== Set default temp_dbg status to 0 so timestep doesn't get cut when heat isn't being solved
				temp_dbg = 0;
				temp_status = 0;
				//== Update thermal profile if new time - old time is bigger than dt_thermal and newton method returns success
				if (((t_mang.get_current_time() + t_mang.get_timestep() - t_thermal_update) >= (paras.temporal.dt_thermal)) && (t_status == timestep_manager_status::SUCCESS))
				{

					// cudaEventRecord(start);
					//== Solve thermal problem
					//-- If successfully solved, temp_dbg = 0
					//-- If error too large, temp_dbg = 1
					update_temperature(t_mang.get_current_time() + t_mang.get_timestep(), t_mang.get_current_time() + t_mang.get_timestep() - t_thermal_update, dev_res_prev, u_ws.h->data_device, dev_res_trans, dev_res_nxt_trans
					, res_temp, dev_res_temp, dev_res_temp_trans, dev_res_temp_prev, dev_res_temp_trans_prev, dev_Tsub, dev_Tsub_prev, dev_T_int, dev_T_int_prev, paras.model, paras.spatial, dev_courant, dev_err_local, temp_dbg 
					, dev_T_int_corrector, dev_matx_sub, dev_matx_main, dev_matx_sup, dev_vecx, dev_maty_sub, dev_maty_main, dev_maty_sup, dev_vecy, dz_A, dz_B, dz_C, dz_D, dev_muzA, dev_muzB, dev_muzC, dev_muzD, dzs_var, dev_T_int_coef
					, h_max, d_max, d_mutex, partial_T_avg, dev_partial_T_avg, dev_vecx_stored, dev_matx_sup_stored, dev_maty_sup_stored, Qterm_stored, dev_Tsub_int1, dev_Tsub_int2
					, dev_Tsub_ikj, dev_Tsub_jki, dev_Tsub_int1_jki, dev_Tsub_int2_kij, Az_ijk_stored, Az_kij_stored, Ay_jki_stored, Ay_ijk_stored);

					//== Error in heat solver too large, temp_dbg = 1
					if (temp_dbg == 1)
					{
						
						//== Step 1: Change Newton Status to something other than newton_status:SUCCESS
						//         In this instance, the film solver will recompute with dt decreased.
						n_status = newton_status::TEMP_CHANGE_DT;
						t_status = t_mang.update_dt(n_status);

						printf("Temp_change_dt!\n");
						
						//== Step 2: Reset film thickness to previous value if film evolution on
						if (paras.model.film_evolution)
						{
							reset_h<<<nB_h, tPB_h>>>(dev_res_prev, u_ws.h->data_device, dims.n_pad, dims.m_pad);
						}

					}
					else 
					{
						//== Heat successfully solved, temp_dbg = 0
						//== If film evolving, update surface tension, gamma 
						//-- Note: probably should do this on device and not host.
						if (paras.model.film_evolution)
						{

							//== Update surface tension, gamma on device. Note that for the Marangoni term we only need gradients in x, y directions
							//== This form is used for the potential to do concentration driven Marangoni effect

							update_surf_tens<<<num_blocks_G, tpb_G>>>(u_ws.gamma->data_device, dev_res_temp, dims.n_pad, dims.padding, dims.n, dims.m 
							, paras.model.gamma_ratio, paras.model.Temp_melt);
							//======================================================//
							//=========================================================//	
						}


						if(paras.model.spat_varying_visc)
						{
							update_visc_rat<<<num_blocks_G, tpb_G>>>(u_ws.visc_rat->data_device, dev_res_temp, dims.n_pad, dims.padding, dims.n, dims.m, paras.model.E_act, paras.model.R, paras.model.T_scl, paras.model.Temp_melt);
						}
						//== Set next time that heat code should be updated
						t_thermal_update = t_mang.get_current_time()+t_mang.get_timestep();

					}

					//== What are we using temp_status for?
					temp_status = 1;

				}

				//== Output all timestep changes if full_text_output is on
				if (paras.io.is_full_text_output)
					output_all_timestep_changes<DATATYPE>(t_mang, n_status, paras.io.is_console_output);

				//== Update dt
				t_status = t_mang.update_dt(n_status);

				//== Add to iteration to log file
				log_file.add_entry(t_mang.get_iteration_index(), t_mang.get_timestep(), n_status, newton_count);
				
			}

			//== End while loop. Film and Heat solver completed successfully

			//== Check if melting mode is turned on

			if (paras.model.melting_switch)		
			{

				int blockSize = 64;
				int gridSize = 64;
				find_minimum_kernel<<<gridSize, blockSize>>>(dev_res_temp, d_min, d_mutex_min, dims.n*dims.m);
				cudaMemcpy(h_min, d_min, sizeof(double), cudaMemcpyDeviceToHost);
				min_temp = (*h_min);
				*h_min = 1000.0;
				cudaMemcpy(d_min, h_min, sizeof(double), cudaMemcpyHostToDevice);
			
				//== Minimum temperature reaches 99.9% melting
				if(min_temp >= 0.999*paras.model.Temp_melt && adapt_starter <= 1){
					if(isFirstRun)
					{
						//== Reset maximum timestep
						paras.temporal.dt_max = 0.01;
						t_mang.update_dt_max(paras.temporal.dt_max);
						t_mang.dt = paras.temporal.dt_max;
						printf("Reached 99.9% to Melting Temperature at t = %f \n", t_mang.get_current_time());
						printf("dt = %f \n", t_mang.get_timestep());
						printf("dt_max = %f \n", paras.temporal.dt_max);
					}
					paras.model.temp_adapt = true;
					adapt_starter += 1;
				}

				//== If the minimum temperature is above the melting threshold turn on film evolution
				if (min_temp >= (paras.model.Temp_melt))
				{
					paras.model.film_evolution = true;

					if (melt_iter == 1 && isFirstRun)
					{
						char buff[100];

						sprintf(buff, "Reached Melting Temperature at t = %f ", t_mang.get_current_time());
						outputString = get_time_stamp() + buff;
						write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString, paras.io.is_console_output);

						melt_iter += 1;						// Increment melt_iter to ensure this change only happens once
						paras.temporal.dt_max = 0.01;
						t_mang.update_dt_max(paras.temporal.dt_max);
					}
				} else if(melt_iter > 1){
					//== Record the solidification time and turn off film evolution.
					paras.model.film_evolution = false;
					if(solid_iter == 1){

						char buff_solid[100];
						sprintf(buff_solid, "Film solidified at t = %f ", t_mang.get_current_time());
						outputString = get_time_stamp() + buff_solid;
						write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString, paras.io.is_console_output);
						solid_iter += 1;
					}
				}

			}

			//== Check adaptive time-stepping is working
			if ( t_status != timestep_manager_status::SUCCESS )
			{
				char buff[100];			

				switch( t_status )
				{
				case timestep_manager_status::MIN_DT:	
					sprintf(buff,"Simulation failed! Timestep below minimum threshold ,dt = %11.10E." , t_mang.get_timestep() );	
					break;
				case timestep_manager_status::DT_CHANGE_OUTPUT:	
					sprintf(buff,"Unexpected State! Newton iteration failed on lowering dt to match time output ,dt = %11.10E." , t_mang.get_timestep() );	
					break;
				default:
					sprintf(buff,"Unhandled timestep_manager_status." );	
					break;

				}
				outputString =  get_time_stamp() + buff;
				write_to_old_file( paras.io.root_directory + file_directories::statusData , outputString , paras.io.is_console_output);
				break;
			}

			//== Output on successful output step
			if ( t_mang.is_sucessful_output_step() )
			{
				char buff[100];		
				
				// Outputting to status file
				sprintf(buff,"Saving solution at t = %11.10E to file." , t_mang.get_current_time() );
				outputString =  get_time_stamp() + buff;
				write_to_old_file( paras.io.root_directory +  file_directories::statusData , outputString, paras.io.is_console_output);

				//sprintf(buff, "Min Temp = %f ", min_temp);
				//outputString = get_time_stamp() + buff;
				//write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString, paras.io.is_console_output);
				
				// Outputting to solution file
				std::string outputFileDir;

				sprintf(buff, "/solution_%07d.bin", t_mang.get_next_output_index() );
				outputFileDir = paras.io.root_directory + file_directories::outputDir + buff;
				memory_manager::copyDeviceToHost<DATATYPE>( u_ws.h );
				output_binary( outputFileDir , u_ws.h->data_host , dims.n_pad , dims.m_pad );

				// sprintf(buff, "/gamma_%07d.bin", t_mang.get_next_output_index());
				// outputFileDir = paras.io.root_directory + file_directories::outputDir + buff;
				// memory_manager::copyDeviceToHost<DATATYPE>( u_ws.gamma );
				// output_binary(outputFileDir, u_ws.gamma->data_host, dims.n_pad, dims.m_pad);

				sprintf(buff2, "/filmtemp_%07d.bin", t_mang.get_next_output_index());
				cudaMemcpy(res_temp, dev_res_temp, (dims.n * dims.m)*sizeof(double), cudaMemcpyDeviceToHost);
				outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
				output_binary(outputFileDir2, res_temp, dims.n, dims.m);

				// sprintf(buff2, "/T_int_%07d.bin", t_mang.get_next_output_index());
				// outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
				// output_binary(outputFileDir2, T_int, dims.n, dims.m);

				if(paras.model.output_sub_temp){
					sprintf(buff2, "/Tsub_%07d.bin", t_mang.get_next_output_index());
					cudaMemcpy(Tsub, dev_Tsub, (dims.n * dims.m)*(dims.p)*sizeof(double), cudaMemcpyDeviceToHost);
					outputFileDir2 = paras.io.root_directory + file_directories::outputDir + buff2;
					output_binary(outputFileDir2, Tsub, dims.n*dims.m*dims.p , 1);
				}

				// cudaMemcpy(Tsub, dev_Tsub, (dims.n * dims.m)*(dims.p)*sizeof(double), cudaMemcpyDeviceToHost);
				// printf("Tsub[n/2][m/2][1]=%f\n", Tsub[1+dims.n/2*dims.p+dims.m/2*dims.n*dims.p]);
				// cudaMemcpy(u_ws.visc_rat->data_host, u_ws.visc_rat->data_device, sizeof(double), cudaMemcpyDeviceToHost);
				// memory_manager::copyDeviceToHost<DATATYPE>( u_ws.visc_rat );
				// printf("u_ws.visc_rat[dims.m/2]= %f\n", u_ws.visc_rat->data_host[((dims.n+4)/2)+(dims.m+4)/2*dims.n]);
			}

			if ( sol_b_mang.is_backup_time() )
			{
				char buff[100];		
				
				sprintf(buff,"Backing up solution at t = %11.10E to file." , t_mang.get_current_time() );
				outputString =  get_time_stamp() + buff;
				write_to_old_file(  paras.io.root_directory + file_directories::statusData , outputString, paras.io.is_console_output);

				memory_manager::copyDeviceToHost<DATATYPE>( u_ws.h );
				output_binary(  paras.io.root_directory + file_directories::backupSolution , u_ws.h->data_host , dims.n_pad , dims.m_pad );

				cudaMemcpy(res_temp, dev_res_temp, (dims.n*dims.m)*sizeof(double), cudaMemcpyDeviceToHost);					// Store film temperature on host
				cudaMemcpy(Tsub, dev_Tsub, (dims.n*dims.m*dims.p)*sizeof(double), cudaMemcpyDeviceToHost);					// Store Tsub on host

				output_binary(  paras.io.root_directory + file_directories::backupfilmtemp , res_temp , dims.n , dims.m );
				output_binary(  paras.io.root_directory + file_directories::backupsubtemp , Tsub , dims.n * dims.m , dims.p );

				save_object<timestep_manager<DATATYPE>>( paras.io.root_directory + file_directories::backupFileInfo , t_mang );

				log_file.commit_data_to_files(paras.io.root_directory);
			}

		}
		//== Loops till t_end reached or break statement is executed by a failure state.

		//== Backup at last time step
		char buff[100];

		sprintf(buff, "Backing up solution at t = %11.10E to file.", t_mang.get_current_time());
		outputString = get_time_stamp() + buff;
		write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString, paras.io.is_console_output);

		memory_manager::copyDeviceToHost<DATATYPE>(u_ws.h);
		output_binary(paras.io.root_directory + file_directories::backupSolution, u_ws.h->data_host, dims.n_pad, dims.m_pad);

		cudaMemcpy(res_temp, dev_res_temp, (dims.n*dims.m)*sizeof(double), cudaMemcpyDeviceToHost);					// Store film temperature on host
		cudaMemcpy(Tsub, dev_Tsub, (dims.n*dims.m*dims.p)*sizeof(double), cudaMemcpyDeviceToHost);					// Store Tsub on host

		output_binary(  paras.io.root_directory + file_directories::backupfilmtemp , res_temp , dims.n , dims.m );
		output_binary(  paras.io.root_directory + file_directories::backupsubtemp , Tsub , dims.n * dims.m , dims.p );

		save_object<timestep_manager<DATATYPE>>(paras.io.root_directory + file_directories::backupFileInfo, t_mang);

		log_file.commit_data_to_files(paras.io.root_directory);

		cudaFree(dev_res_temp);
		cudaFree(dev_res_temp_trans);
		cudaFree(dev_res_temp_prev);
		cudaFree(dev_res_temp_trans_prev); 
		cudaFree(dev_Tsub);
		cudaFree(dev_T_int);
		cudaFree(dev_T_int_prev);
		cudaFree(dev_res_prev);
		cudaFree(dev_res_nxt);
		cudaFree(dev_courant);
		cudaFree(dev_res_trans);
		cudaFree(dev_res_nxt_trans);
		cudaFree(dev_err_local);
		cudaFree(dev_T_int_corrector);
		cudaFree(dev_matx_sub);
		cudaFree(dev_matx_main);
		cudaFree(dev_matx_sup);
		cudaFree(dev_vecx);
		cudaFree(dev_maty_sub);
		cudaFree(dev_maty_main);
		cudaFree(dev_maty_sup);
		cudaFree(dev_vecy);
		cudaFree(dev_muzA);
		cudaFree(dev_muzB);
		cudaFree(dev_muzC);
		cudaFree(dev_muzD);
		cudaFree(dev_T_int_coef);
		cudaFree(d_max);
		cudaFree(d_mutex);
		cudaFree(dev_partial_T_avg);
		cudaFree(d_min);
		cudaFree(d_mutex_min);
		cudaFree(dev_vecx_stored);
		cudaFree(dev_matx_sup_stored);
		cudaFree(dev_maty_sup_stored);
		cudaFree(Qterm_stored);
		cudaFree(dev_Tsub_int1);
		cudaFree(dev_Tsub_int2);
		cudaFree(dev_Tsub_ikj);
		cudaFree(dev_Tsub_jki);
		cudaFree(dev_Tsub_int1_jki);
		cudaFree(dev_Tsub_int2_kij);
		cudaFree(Az_ijk_stored);
		cudaFree(Az_kij_stored);
		cudaFree(Ay_jki_stored);
		// cudaFree(dev_Tzz_stored);
		// cudaFree(dev_Tyy_stored);
		free(h_max);
    	free(partial_T_avg);
		free(h_min);	
		delete[] res_temp;
		delete[] T_int;
		delete[] Tsub;
		delete[] dzs_var;
		delete[] dz_A, dz_B, dz_C, dz_D;
		delete[] T_int_coef;

	};

	void clean_workspace()
	{
		u_ws.clean_workspace();
	}



private:
		
	void setup_partition_and_initialize_memory(DATATYPE x_0, DATATYPE x_n, int n,
		DATATYPE y_0, DATATYPE y_m, int m)
	{
		DATATYPE ds = (x_n - x_0) / (1.0 * (double)n);



	};

	template <typename DATATYPE2> void output_all_timestep_changes(timestep_manager<DATATYPE2> t_mang, newton_status::status n_status, bool  is_console_output)
	{

		string outputString;
		char buff[100];
		sprintf(buff, "dt = %11.10E , t = %11.10E", t_mang.get_timestep(), t_mang.get_current_time());
		outputString = get_time_stamp() + buff;
		write_to_old_file(file_directories::statusData, outputString, is_console_output);

		if (n_status != newton_status::SUCCESS)
		{
			switch (n_status)
			{
			case newton_status::INCREASE_DT:
				// Value should not be returned by newton solver. Dummy case to remove from default case.
				// See 'newton_status' enum for further details.
				break;
			case newton_status::CONVERGENCE_FAILURE_LARGE:
				sprintf(buff, "Newton Failed, dt = %11.10E , t = %11.10E.", t_mang.get_timestep(), t_mang.get_current_time());
				outputString = get_time_stamp() + buff;
				write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString , paras.io.is_console_output );
				break;
			case newton_status::TRUNCATION_ERROR:
			sprintf(buff, "Newton Failed due to Trunc. Err, dt = %11.10E , t = %11.10E.", t_mang.get_timestep(), t_mang.get_current_time());
			outputString = get_time_stamp() + buff;
			write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString , paras.io.is_console_output );
			break;

			}

		}

	}
	string get_time_stamp()
	{
		string time_stamp;
		
		std::chrono::system_clock::time_point p = std::chrono::system_clock::now();
		std::time_t t = std::chrono::system_clock::to_time_t(p);
		time_stamp = std::ctime(&t);
		time_stamp.replace( time_stamp.end()-1 , time_stamp.end() , ":" );
		time_stamp += " ";
		return time_stamp;

	}
	
	dimensions dims;
	parameters<DATATYPE,MODEL_ID,IC_ID> paras;
	unified_work_space<DATATYPE> u_ws;
	cuda_parameters::kernal_launch_parameters ker_launch_paras;

};





#endif

