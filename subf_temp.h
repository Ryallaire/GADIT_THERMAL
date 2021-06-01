// ----------------------------------------------------------------------------------
// Copyright 2016-2017 Michael-Angelo Yick-Hang Lam
//
// The development of this software was supported by the National Science Foundation
// (NSF) Grant Number DMS-1211713.
//
// This file is part of GADIT and was created by Ryan H. Allaire
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
// Name:		subf_temp.h
// Version: 	1.0
// Purpose:		Solves for local temperature T(x,y,z,t) in Film & Substrate Domains.
// CUDA Info:	Program, runs on GPU. Solution for film thickness, h, at time t_{n+1}
//				is inputted as res. Once heat solution is complete, must update material parameters. 
//				This term is then sent to the gpu for solution of thin film equation.
// ----------------------------------------------------------------------------------

#ifndef TEMP_SOLVER
#define TEMP_SOLVER

#include "work_space.h"
#include "parameters.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include "solver_template.h"
#include "change_mat_par.h"
#include <algorithm>

//== Preprocesser numbers for parallelization

//== tPBx*tPBy*tPBz limited due to shared memory; used in substrate solvers
const int tPBx = 28;
const int tPBy = 1;	
const int tPBz = 14;	//== Set this to p-1;

//== Film solver parameters; For full simulataneous solve adix_tpb * adix_numblocks = N
const int adix_tpb = 8;
const int adix_numblocks = 181;
const int adiy_tpb = 8;
const int adiy_numblocks = 181;

//== For sum reduction (TPB must be a power of 2)
const int TPB = 256;
const int BPG = min( 32, (181*181+TPB-1) / TPB );

//== Used for full heat equation in the substrate, if used
const int adiz1_s_mem = 181*1;
const int adiz2_s_mem = 181*1;
const int adiz3_s_mem = tPBz*tPBx*tPBy;

//== Number of threads for ADIZ1 that work on shared memory
//====Note: shared memory should be bigger than block size. Limited by # of threads 1024
const int THREADBLOCK_SIZE_X = adiz1_s_mem/2;
const int THREADBLOCK_SIZE_Y = adiz2_s_mem/2;

//===Declaration of Functions==============================================================================//
__global__ void revert_Tsub_to_prev_val(double* Tsub, double* Tsub_prev, int n, int m, int p);

__global__ void revert_vars_to_prev_val(double* res_temp, double* res_temp_prev, double* res_temp_trans, double* res_temp_trans_prev, double* T_int, double* T_int_prev, double* Tsub, double* Tsub_prev, int n, int m, int p);

__global__ void set_film_sub_interface_temp(double* res_temp, double* Tsub, int n, int m, int p);

__device__ double Q_bar(double hfilm, double t_in, model_parameters<double> &m_paras, double inv_h, double F_of_t);

__device__ void get_mat_adix(double dt_temp_evl, double* res, double* res_nxt, double mu_x, double* mat_sub, double* mat_main, double* mat_sup, int n, int m, int n_pad, int padding, int j, int i, double lambda_x
	, model_parameters<double> m_paras, spatial_parameters<double> &s_paras, double &hx0, double &hxn, int ind_start);

__device__ void get_vec_adix(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_temp, 
	double* T_int, double* T_int_corrector, int j, int i, double courant[], double* vecx, int n, int m, int m_pad, int padding, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, bool is_predict,
	int ind_start, double* vecx_stored, double F_of_t, double F_of_t_prev, double hx0, double hxn);

__global__ void adix(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* T_int, double* T_int_corrector, double courant[], double vecx[]
	, double* matx_sub, double* matx_main, double* matx_sup, int n, int m, int padding, int n_pad, int m_pad, model_parameters<double> m_paras, spatial_parameters<double> s_paras, bool is_predict
	, double* vecx_stored, double* matx_sup_stored, double F_of_t, double F_of_t_prev);

__global__ void adix_solve(double* res_temp, double* matx_sub, double* matx_main, double* matx_sup, double* vecx, int n, int m);

__device__ void get_mat_adiy(double dt_temp_evl,double* res_nxt, double courant[], double* maty_sub, double* maty_main, double* maty_sup, int n, int m, int m_pad, int padding, int i, int j, model_parameters<double> m_paras, spatial_parameters<double> s_paras, double &hx0, double &hxm, int ind_start);

__device__ void get_vec_adiy(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_temp, int i, int j, double* T_int, double courant[], double vecy[], int n, int m, int n_pad, int padding, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, 
	int ind_start, bool is_predict, double* Qterm_stored, double F_of_t, double F_of_t_prev, double* T_int_corrector, double hx0, double hxm);

__global__ void adiy(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* T_int, double courant[], double vecy[]
	, double* maty_sub, double* maty_main, double* maty_sup, int n, int m, int padding, int n_pad, int m_pad, model_parameters<double> m_paras, spatial_parameters<double> s_paras
	, bool is_predict, double* dev_maty_sup_stored, double* Qterm_stored, double F_of_t, double F_of_t_prev, double* T_int_corrector);

__global__ void adiy_solve(double* res_temp_trans, double* maty_sub, double* maty_main, double* maty_sup, double* vecy, int n, int m);

__global__ void transpose_arrays(double* A, double* A_trans, double* B, double* B_trans, double* C, double* C_trans, int n, int m, int n_pad, int m_pad, int padding);

__global__ void reverse_transpose(double* A_trans, double* A, int n, int m);

void update_film(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* courant, double* T_int, double* T_int_corrector, int n, int m, int n_pad, int m_pad, int padding
, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, bool is_predict, double* dev_matx_sub, double* dev_matx_main, double* dev_matx_sup, double* dev_vecx, double* dev_maty_sub, double* dev_maty_main, double* dev_maty_sup, double* dev_vecy
, double* dev_vecx_stored, double* dev_matx_sup_stored, double* dev_maty_sup_stored, double* Qterm_stored, double F_of_t, double F_of_t_prev);

__device__ void get_vec_Crank_z(double dt_temp_evl, double* Tsub, int idx, int idy, double vecz[], double mu_z, int n, int m, int p, double* res_temp, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, int ind_start
, double* muzA, double* muzB, double* muzC, double Bi_new);

__global__ void Crank_z(double dt_temp_evl, double* res_temp, double* Tsub, double mu_z, int n, int m, int p, model_parameters<double> m_paras, spatial_parameters<double> s_paras
, double* muzA, double* muzB, double* muzC, double Bi_new);

__device__ void get_mat_Crank_z(double dt_temp_evl, double* mat_sub, double* mat_main, double* mat_sup, double mu_z, int p, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, int ind_start
, double* muzA, double* muzB, double* muzC, double Bi_new);

void update_sub_temp(double dt_temp_evl, double* res_temp, double* Tsub, double mu_z, int n, int m, int p, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras
, double* dev_muzA, double* dev_muzB, double* dev_muzC, double* dev_muzD, double Bi_new, double* Tsub_int1, double* Tsub_int2, double mu_x, double mu_y
, double* Tsub_ikj, double* Tsub_jki, double* Tsub_int1_jki, double* Tsub_int2_kij, double* Az_ijk_stored, double* Az_kij_stored, double* Ay_jki_stored, double* Ay_ijk_stored);

__global__ void update_substrate_interface(double* Tsub, double* T_int, double inv_dzst2, int n, int m, int p, double* T_int_coef, model_parameters<double> m_paras);

__global__ void store_vars_at_prev_step(double* res_temp, double* res_temp_prev, double* res_temp_trans, double* res_temp_trans_prev, double* T_int, double* T_int_prev, double* Tsub, double* Tsub_prev, int n, int m, int p);

__global__ void store_Tsub_at_prev_step(double* Tsub, double* Tsub_prev, int n, int m, int p);

__device__ double RHS_func(int i, int j, double* T_in, double* T_in_trans, double* h_in, double* h_in_trans, double* T_int_in, double time_in
    , model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, int n, int m, int n_pad, int m_pad, int padding, double F_of_t);

__global__ void compute_err_local(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* res_temp_prev
    , double* res_temp_trans_prev, double* T_int_prev, double* T_int, double* err_local, model_parameters<double> m_paras, spatial_parameters<double> s_paras, int n, int m, int n_pad, int m_pad, int padding
	, double F_of_t, double F_of_t_prev);

void check_local_trunc_err(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp_prev, double* res_temp, double* res_temp_trans, double* res_temp_trans_prev
    , double* T_int_prev, double* T_int, int &temp_dbg, double avg_temp, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras
    , int n, int m, int n_pad, int m_pad, int padding, double* err_local, double* h_max, double* d_max, int* d_mutex, double F_of_t, double F_of_t_prev);

__global__ void find_maximum_kernel(double *array, double *max, int *mutex, unsigned int n);

__global__ void sum_reduction( double *a, double *c, int a_len );

__global__ void update_res_temp(double* vecx, double* res_temp, int n, int m);

__global__ void update_res_temp_trans(double* vecy, double* res_temp_trans, int n, int m);

__device__ void get_TC_funcs(double T, double* TC, double* dTC, double* ddTC, double Temp_scl, int TC_model);

__global__ void Newton_Crank_z(double dt_temp_evl, double* Tsub, double* res_temp, int n, int m, int p, double* dz_A, double* dz_B, double* dz_C, double* dz_D, double Bi_new, model_parameters<double> m_paras
, double D_bdry_term, double E_bdry_term);

__device__ void get_Jacobian(double* Jzm, double* Jz0, double* Jzp, double* T_guess, double T_top, double* A, double* B, double* C, double* D,
		double* dA, double* dB, double* dC, double* dD, int p, double Bi_new, int ind_start, double D_term, double E_term, double Temp_amb, double Temp_scl, int TC_model);

__device__ void get_Newton_LHS(double dt, double* Jzm, double* Jz0, double* Jzp, int p, int ind_start);

__device__ void get_thermal_coef(double* Temp_in, double* A, double* B, double* C, double* D, double* dA, double* dB, double* dC, double* dD, int p, double Temp_scl, double* dz_A, double* dz_B, double* dz_C, double* dz_D, double Bi_new, int ind_start, int TC_model);

__device__ void get_Newton_func(double* func, double* T_in, double T_top, double* A, double* B, double* C, double* D, double Bi_new, int p, int ind_start, double D_term, double E_term, double Temp_amb, double Temp_scl, int TC_model);

__global__ void ADIZ1(double* Tsub, double* Tsub_int1, double* res_temp, int n, int m, int p, double mu_x, double mu_y, double Temp_amb, double Bi_new, double* muzA, double* muzB, double* muzC
, double* Tsub_ikj, double* Tsub_jki, double* Az_ijk_stored, double* Ay_jki_stored, double D_term, double E_term, model_parameters<double> m_paras);

__global__ void ADIZ2(double* Tsub_int1, double* Tsub_int2, double* Tsub, double* res_temp, int n, int m, int p, double mu_y, double* Ay_jki_stored, model_parameters<double> m_paras);

__global__ void ADIZ3(double* Tsub_int2, double* Tsub, double* res_temp, int n, int m, int p, double* muzA, double* muzB, double* muzC, double Temp_amb, double Bi_new
, double* Az_kij_stored, double D_term, double E_term);

__device__ void get_mat_adizx(double* mat_sub, double* mat_main, double* mat_sup, double mu_in, int ind_start, int n_end, int i, double D_SW, double D_SE);

__device__ void get_vec_adizx(double* Tsub, double* res_temp, double* vecz, int ind_start, int i, int j, int k, int n, int m, int p, double Temp_amb, double Bi_new, double mu_x, double mu_y, double* muzA, double* muzB, double* muzC
, double* Tsub_ikj, double* Tsub_jki, double* Az_ijk_stored, double* Ay_ijk_stored, double D_term, double E_term, model_parameters<double> m_paras);

__device__ void get_vec_adizy(double* Tsub_int1, double* Tsub, double* res_temp, double* vecz, int ind_start, int i, int j, int k, int n, int m, int p, double mu_y, double* Ay_jki_stored, model_parameters<double> m_paras);

__device__ void get_mat_adizz(double* mat_sub, double* mat_main, double* mat_sup, int p, int ind_start, double* muzA, double* muzB, double* muzC, double Bi_new, double D_term);

__device__ void get_vec_adizz(double* Tsub_int2, double* Tsub, double* res_temp, double* vecz, int ind_start, int i, int j, int n, int m, int p, double Temp_amb, double Bi_new, double* muzA, double* muzB, double* muzC
, double* Az_kij_stored, double E_term);

__device__ double DXX(double* Tsub, int i, int j, int k, int n, int p, double mu_x, model_parameters<double> m_paras);

__device__ double DYY(double* Tsub, int i, int j, int k, int n, int m, int p, double mu_y, model_parameters<double> m_paras);

__device__ double DZZ(double* Tsub, double* res_temp, int i, int j, int k, int n, int p, double Temp_amb, double Bi_new, double* muzA, double* muzB, double* muzC, double D_term, double E_term);

__global__ void transpose_sub_1(double* Tsub, double* Tsub_ikj, double* Tsub_jki, int n, int m, int p);

__global__ void transpose_A_mats(double* Az_ijk, double* Az_kij, double* Ay_ijk, double* Ay_jki, int n, int m, int p);

__device__ void sub_newton(double &x_in, double Tpm1, double Bi_new, double Temp_amb, double Temp_scl, int TC_model);

__device__ void newt_G(double Tp, double Tpm1, double &G, double &Gp, double Bi_new, double Temp_amb, double Temp_scl, int TC_model);
//=========================================================================================================// 



void update_temperature(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp
, double* dev_res_temp, double* res_temp_trans, double* res_temp_prev, double* res_temp_trans_prev, double* Tsub, double* Tsub_prev, double* T_int, double* T_int_prev
, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, double* dev_courant, double* err_local, int &temp_dbg, double* T_int_corrector 
, double* dev_matx_sub, double* dev_matx_main, double* dev_matx_sup, double* dev_vecx, double* dev_maty_sub, double* dev_maty_main, double* dev_maty_sup, double* dev_vecy 
, double* dz_A, double* dz_B, double* dz_C, double* dz_D, double* dev_muzA, double* dev_muzB, double* dev_muzC, double* dev_muzD, double* dzs_var, double* dev_T_int_coef, double* h_max, double* d_max, int* d_mutex
, double* partial_T_avg, double* dev_partial_T_avg, double* dev_vecx_stored, double* dev_matx_sup_stored,  double* dev_maty_sup_stored, double* Qterm_stored, double* Tsub_int1, double* Tsub_int2
, double* Tsub_ikj, double* Tsub_jki, double* Tsub_int1_jki, double* Tsub_int2_kij, double* Az_ijk_stored, double* Az_kij_stored, double* Ay_jki_stored, double* Ay_ijk_stored)
{
	//== First define Courant Numbers==============================//
	//-- Note: Courant numbers change with changing dt
	double mu_x =	m_paras.K1 * dt_temp_evl * (s_paras.inv_dx2t2);	
	double mu_y =	m_paras.K1 * dt_temp_evl * (s_paras.inv_dy2t2);	
	double mu_z =	m_paras.K2 * dt_temp_evl * (s_paras.inv_dzs2t2);
	double lambda_x = m_paras.K1 * dt_temp_evl * (s_paras.inv_dxt2)*0.5;	
	double lambda_y = m_paras.K1 * dt_temp_evl * (s_paras.inv_dyt2)*0.5;
	bool is_predict;

	double sub_mu_x = m_paras.K2 * dt_temp_evl * (s_paras.inv_dx2t2);
	double sub_mu_y = m_paras.K2 * dt_temp_evl * (s_paras.inv_dy2t2);
	
	//== All Courant numbers stored in one vector.
	double courant[5];					
	courant[0] = mu_x;
	courant[1] = mu_y;
	courant[2] = mu_z;
	courant[3] = lambda_x;
	courant[4] = lambda_y;
	//int N = s_paras.n;

	//== Allocate memory on stack for mu_zA, mu_zB, mu_zC
	double mu_zA[s_paras.p-1], mu_zB[s_paras.p-1], mu_zC[s_paras.p-1];
	double mu_zD[s_paras.p-1];
	for(int j=0; j<s_paras.p-1; j++)
	{
		if(m_paras.nonlin_sub_heat){
			mu_zA[j] = dz_A[j];
			mu_zB[j] = dz_B[j];
			mu_zC[j] = dz_C[j];
			mu_zD[j] = dz_D[j];
		} else{
			mu_zA[j] = m_paras.K2 * dt_temp_evl * 0.5 * dz_A[j];
			mu_zB[j] = m_paras.K2 * dt_temp_evl * 0.5 * dz_B[j];
			mu_zC[j] = m_paras.K2 * dt_temp_evl * 0.5 * dz_C[j];
			mu_zD[j] = 0.0;	
		}

	}
	double Bi_new = m_paras.Bi * dzs_var[s_paras.p-1] / 2.0;


	//== Memory transfer mu_zA, mu_zB, mu_zC
	cudaMemcpy(dev_muzA, mu_zA, (s_paras.p-1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_muzB, mu_zB, (s_paras.p-1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_muzC, mu_zC, (s_paras.p-1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_muzD, mu_zD, (s_paras.p-1)*sizeof(double), cudaMemcpyHostToDevice);

	double T_avg;
	double t_next_output=ceil(t);
	FILE * cK_file;

	dim3 threadsPerBlock(min(32, (int) s_paras.n), min(32, (int) s_paras.m));
    //dim3 numBlocks(ceil(s_paras.n/threadsPerBlock.x), ceil(s_paras.m/threadsPerBlock.y));
	dim3 numBlocks((s_paras.n+threadsPerBlock.x-1)/threadsPerBlock.x, (s_paras.m+threadsPerBlock.y-1)/threadsPerBlock.y);

	//== Copy courant on host to courant on device
	//-- Note: this may be wasting time. May want to define these in gadit or simply pass through scalars
	cudaMemcpy(dev_courant, courant, (5)*sizeof(double), cudaMemcpyHostToDevice);


	//==================================//
	//=---- Begin Prediction Stage ----=//
	//==================================//
	is_predict = true;

    //== Update Substrate Interface at each (x,y) --> (i,j)
    update_substrate_interface<<<numBlocks, threadsPerBlock>>>(Tsub, T_int, s_paras.inv_dzst2, s_paras.n, s_paras.m, s_paras.p, dev_T_int_coef, m_paras);

	// cudaEventRecord(start);
    //== Transpose data from res_temp into res_temp_trans
	transpose_arrays<<< numBlocks, threadsPerBlock >>>(dev_res_temp, res_temp_trans, res, res_trans, res_nxt, res_nxt_trans, s_paras.n, s_paras.m, s_paras.n_pad, s_paras.m_pad, s_paras.padding);
	cudaDeviceSynchronize();
	// cudaEventRecord(stop);

    //== Store variables at previous timestep for adaptive time stepper check
    store_vars_at_prev_step<<<numBlocks, threadsPerBlock>>>(dev_res_temp, res_temp_prev, res_temp_trans, res_temp_trans_prev, T_int, T_int_prev, Tsub, Tsub_prev, s_paras.n, s_paras.m, s_paras.p);
	cudaDeviceSynchronize();

	//== Note the limitation on threads. Product must be less than 1024.
	dim3 nt_store(min(2, (int) s_paras.n), min(2, (int) s_paras.m), min(10, (int) s_paras.p) );
	dim3 nB_store((s_paras.n+nt_store.x-1)/nt_store.x, (s_paras.m+nt_store.y-1)/nt_store.y,(s_paras.p+nt_store.z-1)/nt_store.z);

	store_Tsub_at_prev_step<<<nB_store , nt_store>>>(Tsub, Tsub_prev, s_paras.n, s_paras.m, s_paras.p);
	cudaDeviceSynchronize();

	double F_of_t = m_paras.Q_coef * exp(-pow((t - m_paras.tp_scaled), 2) * m_paras.inv_sigma2_t2_scaled);
	double F_of_t_prev = m_paras.Q_coef * exp(-pow((t-dt_temp_evl - m_paras.tp_scaled), 2) * m_paras.inv_sigma2_t2_scaled);

	//== Update Film Temperature on Device; Function contains Kernel Calls
    update_film(t, dt_temp_evl, res, res_nxt, res_trans, res_nxt_trans, dev_res_temp, res_temp_trans, dev_courant, T_int, T_int_corrector, s_paras.n, s_paras.m, s_paras.n_pad, s_paras.m_pad, s_paras.padding, m_paras, s_paras, is_predict
	, dev_matx_sub, dev_matx_main, dev_matx_sup, dev_vecx, dev_maty_sub, dev_maty_main, dev_maty_sup, dev_vecy, dev_vecx_stored, dev_matx_sup_stored, dev_maty_sup_stored, Qterm_stored, F_of_t, F_of_t_prev);

    //== Update Substrate Temperature on Device; Function contains Kernel Calls
    update_sub_temp(dt_temp_evl, dev_res_temp, Tsub, mu_z, s_paras.n, s_paras.m, s_paras.p, m_paras, s_paras, dev_muzA, dev_muzB, dev_muzC, dev_muzD, Bi_new, Tsub_int1, Tsub_int2, sub_mu_x, sub_mu_y
	, Tsub_ikj, Tsub_jki, Tsub_int1_jki, Tsub_int2_kij, Az_ijk_stored, Az_kij_stored, Ay_jki_stored, Ay_ijk_stored);

	//== Set Top Layer of Substrate to Metal Temperature
	set_film_sub_interface_temp<<< numBlocks, threadsPerBlock >>>(dev_res_temp, Tsub, s_paras.n, s_paras.m, s_paras.p);
	cudaDeviceSynchronize();


    //== Update Substrate Interface at each (x,y) --> (i,j)
	update_substrate_interface<<<numBlocks, threadsPerBlock>>>(Tsub, T_int_corrector, s_paras.inv_dzst2, s_paras.n, s_paras.m, s_paras.p, dev_T_int_coef, m_paras);
	//cudaDeviceSynchronize();

	//==================================//
	//=---- Begin Correction Stage ----=//
	//==================================//

	is_predict = false;
	
	//== Reset res_temp and Tsub to previous values
	revert_vars_to_prev_val<<<numBlocks, threadsPerBlock>>>(dev_res_temp, res_temp_prev, res_temp_trans, res_temp_trans_prev, T_int, T_int_prev, Tsub, Tsub_prev, s_paras.n, s_paras.m, s_paras.p);

	//== Reset Tsub (faster to do this routine alone)
	revert_Tsub_to_prev_val<<<nB_store, nt_store>>>(Tsub, Tsub_prev, s_paras.n, s_paras.m, s_paras.p);
	cudaDeviceSynchronize();

	//== Update Film Temperature on Device; Function contains Kernel Calls
    update_film(t, dt_temp_evl, res, res_nxt, res_trans, res_nxt_trans, dev_res_temp, res_temp_trans, dev_courant, T_int, T_int_corrector, s_paras.n, s_paras.m, s_paras.n_pad, s_paras.m_pad, s_paras.padding, m_paras, s_paras, is_predict
	, dev_matx_sub, dev_matx_main, dev_matx_sup, dev_vecx, dev_maty_sub, dev_maty_main, dev_maty_sup, dev_vecy, dev_vecx_stored, dev_matx_sup_stored, dev_maty_sup_stored, Qterm_stored, F_of_t, F_of_t_prev);

    //== Update Substrate Temperature on Device; Function contains Kernel Calls
    update_sub_temp(dt_temp_evl, dev_res_temp, Tsub, mu_z, s_paras.n, s_paras.m, s_paras.p, m_paras, s_paras, dev_muzA, dev_muzB, dev_muzC, dev_muzD, Bi_new, Tsub_int1, Tsub_int2, sub_mu_x, sub_mu_y
	, Tsub_ikj, Tsub_jki, Tsub_int1_jki, Tsub_int2_kij, Az_ijk_stored, Az_kij_stored, Ay_jki_stored, Ay_ijk_stored);

	//== Set Top Layer of Substrate to Metal Temperature
	set_film_sub_interface_temp<<< numBlocks, threadsPerBlock >>>(dev_res_temp, Tsub, s_paras.n, s_paras.m, s_paras.p);

    //== Update Substrate Interface at each (x,y) --> (i,j)
	update_substrate_interface<<<numBlocks, threadsPerBlock>>>(Tsub, T_int, s_paras.inv_dzst2, s_paras.n, s_paras.m, s_paras.p, dev_T_int_coef, m_paras);

	//==========================================================//
	//=---- Solution completed prediction/correction stage ----=//
	//==========================================================//


	//== Compute Average film temperature at time t^{n+1}
	//== Compute average on GPU
	//== Step1: Sum the array in blocks
	sum_reduction<<<BPG, TPB>>>(dev_res_temp, dev_partial_T_avg, s_paras.n*s_paras.m);
	cudaMemcpy( partial_T_avg, dev_partial_T_avg, BPG*sizeof(double), cudaMemcpyDeviceToHost );

    T_avg = 0;
    for (int i=0; i<BPG; i++) {
        T_avg += partial_T_avg[i];
    }
	T_avg /= (s_paras.n * s_paras.m);

	//== Check if adaptive time stepping is turned on for heat code
	if (m_paras.temp_adapt) 
	{

		//== Check the value of the maximum relative local truncation error
		//-- Note: if the error is larger than temp_tol temp_dbg will return 1; else return 0
		check_local_trunc_err(t, dt_temp_evl, res, res_nxt, res_trans, res_nxt_trans, res_temp_prev, dev_res_temp, res_temp_trans, res_temp_trans_prev
        , T_int_prev, T_int, temp_dbg, T_avg, m_paras, s_paras, s_paras.n, s_paras.m, s_paras.n_pad, s_paras.m_pad, s_paras.padding, err_local, h_max, d_max, d_mutex
		, F_of_t, F_of_t_prev); 

		//== If local truncation error big, reset everything===//
		if (temp_dbg == 1) 
		{
			
			//== Reset all heat variables to previous values
			revert_vars_to_prev_val<<<numBlocks, threadsPerBlock>>>(dev_res_temp, res_temp_prev, res_temp_trans, res_temp_trans_prev, T_int, T_int_prev, Tsub, Tsub_prev, s_paras.n, s_paras.m, s_paras.p);
			revert_Tsub_to_prev_val<<<nB_store, nt_store>>>(Tsub, Tsub_prev, s_paras.n, s_paras.m, s_paras.p);
			cudaMemcpy(res_temp, dev_res_temp, (s_paras.n * s_paras.m)*sizeof(double), cudaMemcpyDeviceToHost);

		}
		else {

			//== Error small, accept values of variables at time t^{n+1}

			//== Change material parameters based on average temperature
			change_mat_par(T_avg, m_paras);

			//== Write the values of the parameters to a file
			if (t >= t_next_output) {
				if (t_next_output == 1.0) {
					cK_file = fopen("cK_file.dat", "w");
					fprintf(cK_file, "%s\t %s\t %s\n", "t", "cK", "T_avg");
					fprintf(cK_file, "%f\t %f\t %f\n", t, m_paras.cK, T_avg);
					fclose(cK_file);
				}
				else {
					cK_file = fopen("cK_file.dat", "a");
					fprintf(cK_file, "%f\t %f\t %f\n", t, m_paras.cK, T_avg);
					fclose(cK_file);
					//t_next_output = t_next_output + 1.0;
				}

			}
		}
	}
	else {

		//== If adaptive time stepping is turned off, automatically set temp_dbg=0
		//==   so that the error is always accepted as small.
		temp_dbg = 0;

	}
	//===========================================================//

	//End of Temperature update
	//Proceed to send temperature information to gamma (surf tens.) and output file//
}


__device__ void tri_solve(double* mat_sub, double* mat_main, double* mat_sup , double* vec, int n, int l_mat, int l_vec)
{	
	//==Description: Tri-diagonal Thomas algorithm solver
	n--;

    // Saving previous c and d to be used later as to avoid rereading from global memory
	double c_im = mat_sup[l_mat + 0] / mat_main[l_mat + 0];
	double d_im = vec[l_vec + 0] / mat_main[l_mat + 0];

	//== Register variables to take from global memory
	double a_i;
	double b_i;
	double temp_inv;

	double c_i;
	double d_i;

	double c_ip;
	double d_ip;

	mat_sup[l_mat + 0] = c_im;
	vec[l_vec + 0] = d_im;


	for(int i=1; i<n; i++){

		a_i = mat_sub[l_mat+i];
		b_i = mat_main[l_mat+i];
		temp_inv = 1.0/(b_i - a_i * c_im);

		// Saving previous c and d to be used later as to avoid rereading from global memory
		c_im = mat_sup[l_mat + i] * (temp_inv);
		d_im = (vec[l_vec + i] - a_i * d_im) * (temp_inv);

		// Saving to global memory
		mat_sup[l_mat+i] = c_im;			
		vec[l_vec + i] = d_im;
	}
    //== Solve for last x value
	a_i = mat_sub[l_mat+n];
	b_i = mat_main[l_mat+n];
	temp_inv = 1.0/(b_i - a_i* c_im);
	vec[l_vec+n] = (vec[l_vec+n] - a_i * d_im) * (temp_inv);

    //== Solve for remaining x values by back substitution
	d_ip = vec[l_vec+n+1];
	for(int i=n; i >= 0; i--){
		d_i = vec[l_vec+i] - mat_sup[l_mat+i]*d_ip;
		vec[l_vec+i] = d_i;
		d_ip = d_i;
	}
 
}

__device__ double Q_bar(double hfilm, double t_in, model_parameters<double> &m_paras, double inv_h, double F_of_t)
{
	double Q_avg;
	// Average Uniform Source
	//Q_avg	=	m_paras.zeta*(1.0-exp(-1.0*(hfilm)/m_paras.alpha_m_inv*m_paras.l_scl)) 
	//*(1.0-m_paras.reflect_coef*(1.0-exp(-1.0*hfilm/m_paras.alpha_r_inv*m_paras.l_scl)))/(m_paras.l_scl*hfilm)*m_paras.alpha_m_inv;

	// Average Gaussian Source
	//	Q_avg	=	m_paras.tp*m_paras.zeta/m_paras.sigma*exp(-pow((t_in*m_paras.t_scl-m_paras.tp),2)/2.0/pow(m_paras.sigma,2))
	//		*(1.0-exp(-1.0*(hfilm)/m_paras.alpha_m_inv*m_paras.l_scl)) 
	//	*(1.0-m_paras.reflect_coef*(1.0-exp(-1.0*hfilm/m_paras.alpha_r_inv*m_paras.l_scl)))/(m_paras.l_scl*hfilm)*m_paras.alpha_m_inv;

	//	Q_avg = m_paras.tp*m_paras.zeta / sqrt(2.0*(atan(1)*4.0)) / m_paras.sigma*exp(-pow((t_in*m_paras.t_scl - m_paras.tp), 2) / 2.0 / pow(m_paras.sigma, 2))

	// Q_avg = m_paras.tp*m_paras.zeta / sqrt(2.0*(M_PI*1.0)) / m_paras.sigma*exp(-pow((t_in*m_paras.t_scl - m_paras.tp), 2) / 2.0 / pow(m_paras.sigma, 2))
	// 	*(1.0 - exp(-1.0*(hfilm) / m_paras.alpha_m_inv*m_paras.l_scl))
	// 	*(1.0 - m_paras.reflect_coef*(1.0 - exp(-1.0*hfilm / m_paras.alpha_r_inv*m_paras.l_scl))) / (m_paras.l_scl*hfilm)*m_paras.alpha_m_inv;

	//Q_avg = m_paras.Q_coef //exp(-pow((t_in - m_paras.tp_scaled), 2) * m_paras.inv_sigma2_t2_scaled)
	Q_avg = F_of_t * (1.0 - exp(-1.0*(hfilm) * m_paras.alpha_m_scaled))
	*(1.0 - m_paras.reflect_coef*(1.0 - exp(-1.0*hfilm * m_paras.alpha_r_scaled))) * inv_h; // / (hfilm);

	// Zeroed Source
	//Q_avg = 0.0;

	//== Note: need to make a switch to modulate forms of Q used

	return Q_avg;
}

void update_film(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* courant, double* T_int, double* T_int_corrector, int n, int m, int n_pad, int m_pad, int padding
	, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, bool is_predict, double* dev_matx_sub, double* dev_matx_main, double* dev_matx_sup, double* dev_vecx, double* dev_maty_sub, double* dev_maty_main, double* dev_maty_sup, double* dev_vecy
	, double* dev_vecx_stored, double* dev_matx_sup_stored, double* dev_maty_sup_stored, double* Qterm_stored, double F_of_t, double F_of_t_prev)
{
	//== Initialization of Parameters
	dim3 threadsPerBlock(min(32,n),min(32,m));
	dim3 numBlocks((n+threadsPerBlock.x-1)/threadsPerBlock.x, (m+threadsPerBlock.y-1)/threadsPerBlock.y);

	//==Small Domain threads per block (tPB) for adi schemes for film temperature
	// Note: minimum is used so domain size can be decreased
	dim3 tPB_adix(min(8,m),min(8,n));
	dim3 tPB_adiy(min(8,n),min(8,m));

	//==Large Domain alternatives
	// dim3 tPB_adix(min(32,m),min(32,n));
	// dim3 tPB_adix(min(16,m),min(16,n));

	//== Number of blocks used in adi schemes
	dim3 nB_adix(8, (n+tPB_adix.y-1)/tPB_adix.y);
	dim3 nB_adiy((n+tPB_adiy.x-1)/tPB_adiy.x, (m+tPB_adiy.y-1)/tPB_adiy.y) ;

	//== Solve intermediate T* parallelized over every y step
	adix<<< nB_adix, tPB_adix >>>(t, dt_temp_evl, res, res_nxt, res_trans, res_nxt_trans, res_temp, res_temp_trans, T_int, T_int_corrector, courant, dev_vecx, dev_matx_sub, dev_matx_main, dev_matx_sup, n, m, padding, n_pad
									, m_pad, m_paras, s_paras, is_predict, dev_vecx_stored, dev_matx_sup_stored, F_of_t, F_of_t_prev);
	cudaDeviceSynchronize();

	//== Large Domain
	int nt = min(adix_tpb,m);
	int nB = adix_numblocks;
	
	//== Small Domain
	// int nt = 181;
	// int nB = 1;

	//== Solve Ax=b
	adix_solve<<< nB, nt >>>(res_temp, dev_matx_sub, dev_matx_main, dev_matx_sup, dev_vecx, n, m);
	cudaDeviceSynchronize();

	update_res_temp<<< numBlocks, threadsPerBlock >>>(dev_vecx, res_temp, n, m);
	cudaDeviceSynchronize();

	//== Large Domain
	int nty = min(adiy_tpb,n);
	int nBy = adiy_numblocks;

	//== Solve next time step T^{n+1} parallelized over every x step
	adiy<<< nB_adiy, tPB_adiy >>>(t, dt_temp_evl, res, res_nxt, res_nxt_trans, res_temp, res_temp_trans, T_int, courant, dev_vecy, dev_maty_sub, dev_maty_main, dev_maty_sup, n, m , padding, n_pad, m_pad, m_paras, s_paras
	, is_predict, dev_maty_sup_stored, Qterm_stored, F_of_t, F_of_t_prev, T_int_corrector);
	cudaDeviceSynchronize();

	//== Solve Ay=b
	adiy_solve<<< nBy, nty >>>(res_temp_trans, dev_maty_sub, dev_maty_main, dev_maty_sup, dev_vecy, n, m);
	cudaDeviceSynchronize();

	update_res_temp_trans<<< numBlocks, threadsPerBlock >>>(dev_vecy, res_temp_trans, n, m);

	//== Send data from res_temp_trans to res_temp
	reverse_transpose<<< numBlocks, threadsPerBlock >>>(res_temp_trans, res_temp, n, m);

    //== Free matrix/vector memory from Device
	//== Note: we may want to free up memory in between kernel calls to save memory
    //===========================================================================//
	
}

__global__ void adix(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* T_int, double* T_int_corrector, double courant[]
, double* vecx, double* matx_sub, double* matx_main, double* matx_sup, int n, int m, int padding, int n_pad, int m_pad, model_parameters<double> m_paras, spatial_parameters<double> s_paras, bool is_predict
, double* vecx_stored, double* matx_sup_stored, double F_of_t, double F_of_t_prev)
{
	//int Nx=n; //s_paras.m;
	double mu_x = courant[0];
	double lambda_x = courant[3];

	int idy = threadIdx.x + blockIdx.x * blockDim.x;

	// Add x workers to handle x components of matrix/vector
	int idx = threadIdx.y + blockIdx.y * blockDim.y;
	int ind_start;
	double hx0, hxn;

	if ( idy < m && idx < n )
	{	
		while(idx < n)
		{
			while(idy < m)
			{
		ind_start = idy*n;
		//== Build matrix
			//-- Note: need res since matx uses hx
		if(is_predict){
			get_mat_adix(dt_temp_evl, res, res_nxt, mu_x, matx_sub, matx_main, matx_sup, n, m, n_pad, padding, idy, idx, lambda_x, m_paras, s_paras, hx0, hxn, ind_start);
			matx_sup_stored[ind_start+idx] = matx_sup[ind_start+idx];
		} else {
			matx_sup[ind_start+idx] = matx_sup_stored[ind_start+idx];
		}


		//== Build vector
			//-- Note: need res_temp_trans since vecx using hy and Ty and Tyy
		get_vec_adix(t, dt_temp_evl, res_trans, res_nxt_trans, res_temp_trans, T_int, T_int_corrector, idy, idx, courant, vecx, n, m, m_pad, padding, m_paras, s_paras, is_predict, ind_start, vecx_stored
		, F_of_t, F_of_t_prev, hx0, hxn);

				idy += blockDim.x * gridDim.x;
			}
			idy = threadIdx.x + blockIdx.x * blockDim.x;
			idx += blockDim.y * gridDim.y;
		}
	}
}

__global__ void adix_solve(double* res_temp, double* matx_sub, double* matx_main, double* matx_sup, double* vecx, int n, int m)
{
	int idy = threadIdx.x + blockIdx.x * blockDim.x;
	int ind_start;

	if ( idy < m )
	{	
		while(idy < m)
		{
		 ind_start = idy*n;
		 
		tri_solve(matx_sub, matx_main, matx_sup, vecx, n, ind_start, ind_start);

		idy += blockDim.x * gridDim.x;
		}
	}
}

__global__ void adiy(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* T_int, double courant[], double vecy[]
	, double* maty_sub, double* maty_main, double* maty_sup, int n, int m, int padding, int n_pad, int m_pad, model_parameters<double> m_paras, spatial_parameters<double> s_paras
	, bool is_predict, double* maty_sup_stored, double* Qterm_stored, double F_of_t, double F_of_t_prev, double* T_int_corrector)
{
	//== Function takes in res_temp at intermediate step for fixed x and outputs solution to Ax=b;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int ind_start;
	// Kernel call matrix adiy with 
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	double hx0, hxm;

	if ( idx < n && idy < m )
	{
		ind_start = idx*m;
		//== Build matrix
		//-- Note: need res_trans since maty uses hy
		if(is_predict)
		{
			get_mat_adiy(dt_temp_evl, res_nxt_trans, courant, maty_sub, maty_main, maty_sup, n, m, m_pad, padding, idx, idy, m_paras, s_paras, hx0, hxm, ind_start);
			maty_sup_stored[ind_start+idy] = maty_sup[ind_start+idy];
		} else{
			maty_sup[ind_start+idy] = maty_sup_stored[ind_start+idy];
		}


		//== Build vector
		//-- Note: need variables with indices i+n*j
		get_vec_adiy(t, dt_temp_evl, res, res_nxt, res_temp, idx, idy, T_int, courant, vecy, n, m, n_pad, padding, m_paras, s_paras, ind_start, is_predict, Qterm_stored
		, F_of_t, F_of_t_prev, T_int_corrector, hx0, hxm);

	}


}

__global__ void adiy_solve(double* res_temp_trans, double* maty_sub, double* maty_main, double* maty_sup, double* vecy, int n, int m)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int ind_start;

	//==Note: changed this line to n 7/1/20
	if ( idx < n )
	{
		while(idx < n)
		{
			ind_start = idx*m;
			tri_solve(maty_sub, maty_main, maty_sup, vecy, m, ind_start, ind_start);

		idx += blockDim.x * gridDim.x;
		}
	}
}

__device__ void get_mat_adix(double dt_temp_evl, double* res, double* res_nxt, double mu_x, double* mat_sub, double* mat_main, double* mat_sup, int n, int m, int n_pad, int padding, int j, int i, double lambda_x, model_parameters<double> m_paras, spatial_parameters<double> &s_paras, double &hx0, double &hxn, int ind_start)
{
	int kc, kp, km;
	double hx, hxdbhtl, hx_nxt;

	if(i == 0)
	{
		// Define boundary (i,j)=(0,j) (x=0)
		kc = (j+padding)*n_pad + padding;
		kp = kc + 1;
		// int ind_start = j*n;
		
		// Define (hx/h)^* term
		// Approximate with an average (hx/h)^* = 0.5*[ (hx/h)^{n} + (hx/h)^{n+1} ]
		
		//== Approximating (hx/h)^n (time, n)
		hx = (res[kp] - res[kc])*(s_paras.inv_dxt2);
		hxdbhtl = hx / res[kc] * lambda_x;

		//== Approximating (hx/h)^{n+1} (time, n+1)
		hx_nxt = (res_nxt[kp] - res_nxt[kc])*(s_paras.inv_dxt2);
		hxdbhtl += hx_nxt / res_nxt[kc] * lambda_x;
		hxdbhtl *= 0.5;

		//== Send to output for vecx computation
		hx0 = mu_x-hxdbhtl;

		mat_sub[ind_start+0] = 0.0;
		mat_main[ind_start+0] = 1.0+2.0*mu_x + (-mu_x+hxdbhtl)*m_paras.D_W; // Robin-type boundary condition (previously Neumann = 1+mu_x+hxdbhtl);
		mat_sup[ind_start+0] = -mu_x-hxdbhtl;								// 5/22/20 added 0.5 hxdbhtl

	} else if(i==(n-1)){
		
		kc = (j+padding)*n_pad + (n-1) + padding;
		km = kc - 1; //(j+padding)*n_pad + (n-2)*padding;

		//== Approximating (hx/h)^{n} (time, n)
		hx = (res[kc] - res[km])*(s_paras.inv_dxt2);
		hxdbhtl = hx / res[kc] * lambda_x;

		//== Approximating (hx/h)^{n+1} (time, n+1)
		hx_nxt = (res_nxt[kc] - res_nxt[km])*(s_paras.inv_dxt2);
		hxdbhtl += hx_nxt / res_nxt[kc] * lambda_x;
		hxdbhtl *= 0.5;

		//== Send to output for vecx computation
		hxn = mu_x+hxdbhtl;
										
		mat_sub[ind_start + n-1] = -mu_x+hxdbhtl;
		mat_main[ind_start + n-1] = 1.0+2.0*mu_x - (mu_x+hxdbhtl)*m_paras.D_E; //1+mu_x-hxdbhtl;
		mat_sup[ind_start + n-1] = 0.0;

	} else if(i<n){
    	//for(int i=1;i<=(n-2);i++){
		kc = (j+padding)*n_pad + i + padding;
		kp = kc + 1;
		km = kc - 1;

		//== Approximating (hx/h)^{n} (time, n)
		hx = (res[kp] - res[km])*(s_paras.inv_dxt2);
		hxdbhtl = hx / res[kc] * lambda_x;

		//== Approximating (hx/h)^{n+1} (time, n+1)
		hx_nxt = (res_nxt[kp] - res_nxt[km])*(s_paras.inv_dxt2);
		hxdbhtl += hx_nxt / res_nxt[kc] * lambda_x;
		hxdbhtl *= 0.5;

        mat_sub[ind_start+i] = -mu_x+hxdbhtl;
        mat_main[ind_start+i] = 1.0+2.0*mu_x;
		mat_sup[ind_start+i] = -mu_x-hxdbhtl;


	}

}

__device__ void get_vec_adix(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_temp, 
	double* T_int, double* T_int_corrector, int j, int i, double courant[], double* vecx, int n, int m, int m_pad, int padding, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, bool is_predict,
	int ind_start, double* vecx_stored, double F_of_t, double F_of_t_prev, double hx0, double hxn)
{		
	int kc, kp, km, khc, khp, khm;
	
	double mu_y	= courant[1];
	double lambda_y = courant[4];

	double hy;
	double hydbh;									//hy divided by h
	double C, D;

	//== New register variables
	double inv_res;

	//==Initialize ghost points for boundaries y=0,L, (j=0,m-1)
	double T_ghost0, T_ghostm;

	//== Determine coefficients for T_int term based on prediction/correction step
	if(is_predict)
	{
		C = -2.0*0.5;
		D = 0.0;
	} else{
		C = -1.0*0.5;
		D = -1.0*0.5;
	}
	
	if(i<n){

		kc = i*m + j;
		km = kc - 1;
		kp = kc + 1;
		khc = padding+j + (i+padding)*m_pad;
		khp = khc + 1;
		khm = khc - 1;
		inv_res = 1.0 / res[khc];

		if(is_predict){
			if(j==0){

				hy = (res[khp] - res[khc])*(s_paras.inv_dyt2);
				hydbh = lambda_y * hy * inv_res; // / res[khc];
				T_ghost0 = m_paras.D_S * res_temp[kc] + m_paras.E_S;



				vecx[ind_start+i] = (mu_y - hydbh)*T_ghost0 + (1 - 2 * mu_y)*res_temp[kc] + (mu_y + hydbh)*res_temp[kp] +
					dt_temp_evl * (C*m_paras.K1_t_kratio_db2 * inv_res*T_int[kc] +
					0.25*(Q_bar(res[khc], t - dt_temp_evl, m_paras, inv_res, F_of_t_prev)+ Q_bar(res_nxt[khc], t, m_paras, inv_res, F_of_t)));

			} 
			else if (j == (m-1)){
				hy = (res[khc] - res[khm])*(s_paras.inv_dyt2);
				hydbh = lambda_y * hy * inv_res; // / res[khc];

				T_ghostm = m_paras.D_N * res_temp[kc] + m_paras.E_N;

				//Define vecx for case i, j=m-1====================//
				vecx[ind_start+i] = (mu_y - hydbh)*res_temp[km] + (1 - 2 * mu_y)*res_temp[kc] + (mu_y + hydbh)*T_ghostm+
					dt_temp_evl * (C*m_paras.K1_t_kratio_db2 * inv_res*T_int[kc] +
					0.25*(Q_bar(res[khc], t - dt_temp_evl, m_paras, inv_res, F_of_t_prev)+ Q_bar(res_nxt[khc], t, m_paras, inv_res, F_of_t)));

			} 
			else{
				
				hy = (res[khp] - res[khm])*(s_paras.inv_dyt2);
				hydbh = lambda_y * hy * inv_res; // / res[khc];

				//==Define vecx for i free, j free
				vecx[ind_start+i] = (mu_y - hydbh)*res_temp[km] + (1 - 2 * mu_y)*res_temp[kc] + (mu_y + hydbh)*res_temp[kp] +
					dt_temp_evl * (C*m_paras.K1_t_kratio_db2 * inv_res*T_int[kc] +
						0.25*(Q_bar(res[khc], t - dt_temp_evl, m_paras, inv_res, F_of_t_prev)+ Q_bar(res_nxt[khc], t, m_paras, inv_res, F_of_t))); 
			}
			if(i==0){
				vecx[ind_start+0] += hx0*m_paras.E_W;	//== West boundary x=0
			} else if(i==(n-1)){
				vecx[ind_start+n-1] += hxn*m_paras.E_E;	//== East boundary x=L
			}
			vecx_stored[ind_start + i] = vecx[ind_start + i];
		} else{
			vecx[ind_start + i] = vecx_stored[ind_start + i] + dt_temp_evl * (0.5*m_paras.K1_t_kratio_db2 / (res[khc])*T_int[kc] +
					D*m_paras.K1_t_kratio_db2 / (res_nxt[khc])*T_int_corrector[kc]);
		}

	// }

	}
}

__device__ void get_mat_adiy(double dt_temp_evl,double* res_nxt, double courant[], double* maty_sub, double* maty_main, double* maty_sup, int n, int m, int m_pad, int padding, int i, int j,model_parameters<double> m_paras, spatial_parameters<double> s_paras, double &hx0, double &hxm, int ind_start)
{	
	//Define Data Types of Local Variables
	double mu_y = courant[1];
	double lambda_y = courant[4];
	double hy;
	double hydbhtl;
	int k, kp, km;

	if(j==0){
		// Indices at (i,j) = (i,0)
		k = (i+padding)*m_pad + padding;
		kp = k + 1;

		hy = (res_nxt[kp]-res_nxt[k])*(s_paras.inv_dyt2);
		hydbhtl = hy/res_nxt[k]*lambda_y;

		//== Store for calculation in vector
		hx0 = mu_y-hydbhtl;
		
		maty_sub[ind_start+0] = 0;
		maty_main[ind_start+0] = 1.0+2.0*mu_y + (-mu_y+hydbhtl)*m_paras.D_S;
		maty_sup[ind_start+0] = -mu_y-hydbhtl;

	} else if (j == m-1){
		k = padding+(m-1) + (i+padding)*m_pad;
		km = k-1;

		hy = (res_nxt[k] - res_nxt[km])*(s_paras.inv_dyt2);
		hydbhtl = hy / res_nxt[k] * lambda_y;

		//== Store for calculation in vector
		hx0 = mu_y-hydbhtl;

		maty_sub[ind_start+m-1] = -mu_y+hydbhtl;
		maty_main[ind_start+m-1] = 1.0+2.0*mu_y-(mu_y+hydbhtl)*m_paras.D_N;
		maty_sup[ind_start+m-1] = 0;
	} else if ( j < m){
		k = padding+j + (i+padding)*m_pad;
		kp = k+1;
		km = k-1;
		
		hy = (res_nxt[kp]-res_nxt[km])*(s_paras.inv_dyt2);
		hydbhtl = hy/res_nxt[k] * lambda_y;
		
        maty_sub[ind_start+j] = -mu_y+hydbhtl;
        maty_main[ind_start+j] = 1.0+2.0*mu_y;
        maty_sup[ind_start+j] = -mu_y-hydbhtl;
	}
}

__device__ void get_vec_adiy(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_temp, int i, int j, double* T_int, double courant[], double vecy[], int n, int m, int n_pad, int padding, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras,
int ind_start, bool is_predict, double* Qterm_stored, double F_of_t, double F_of_t_prev, double* T_int_corrector, double hx0, double hxm)
{
	double mu_x = courant[0];
	double lambda_x = courant[3];
	double hx;

	double hxdbhtl;	
	int k, kp, km, kh, khp, khm;

	// Input: 
		// res[k]; k= padding+i + (j+padding)*n_pad
		// res_nxt[k]; padding+i + (j+padding)*n_pad
		// res_temp[k]; i + j*n

		k = i + j*n;
		kp = i + j*n + 1;
		km = i + j*n - 1;
		kh = padding+i + (j+padding)*n_pad;
		double inv_res = 1.0 / res[kh];
		double inv_res_nxt = 1.0 / res_nxt[kh];
		double C, D;
		double hx_nxt, T_ghost0, T_ghostn;

		if(is_predict){
			//== If on prediction phase, store Qterm for later use
			Qterm_stored[ind_start + j] = dt_temp_evl * (0.25*(Q_bar(res_nxt[kh], t, m_paras, inv_res, F_of_t)+ Q_bar(res[kh], t-dt_temp_evl, m_paras, inv_res, F_of_t_prev)));
		}

		//== Determine coefficients for T_int term based on prediction/correction step
		if(is_predict)
		{
			C = -2.0;
			D = 0.0;
		} else{
			C = -1.0;
			D = -1.0;
		}
		
		if(i==0){

			khp = padding+(i+1) + (j+padding)*n_pad;

			//First Define hx at time t_k (approximation made here, instead of t_*)
			hx = (res[khp]-res[kh])*(s_paras.inv_dxt2);
			hxdbhtl = hx * inv_res*lambda_x;

			//== Approximating (hx/h)^{n+1} (time, n+1)
			hx_nxt = (res_nxt[khp] - res_nxt[kh])*(s_paras.inv_dxt2);
			hxdbhtl += hx_nxt * inv_res_nxt * lambda_x;
			hxdbhtl *= 0.5;

			T_ghost0 = m_paras.D_W * res_temp[k] + m_paras.E_W;
			vecy[ind_start+j]=(mu_x-hxdbhtl)*T_ghost0+(1.0-2.0*mu_x)*res_temp[k]+(mu_x+hxdbhtl)*res_temp[kp]+ 
							Qterm_stored[ind_start + j];


		} else if (i==(n-1)){

			khm = padding+(i-1) + (j+padding)*n_pad;

			hx = (res[kh] - res[khm])*(s_paras.inv_dxt2);
			hxdbhtl = hx * inv_res*lambda_x;

			//== Approximating (hx/h)^{n+1} (time, n+1)
			hx_nxt = (res_nxt[kh] - res_nxt[khm])*(s_paras.inv_dxt2);
			hxdbhtl += hx_nxt * inv_res_nxt * lambda_x;
			hxdbhtl *= 0.5;

			T_ghostn = m_paras.D_E * res_temp[k] + m_paras.E_E;
			vecy[ind_start+j]=(mu_x-hxdbhtl)*res_temp[km]+(1-2*mu_x)*res_temp[k]+(mu_x+hxdbhtl)*T_ghostn+ 
						Qterm_stored[ind_start + j];

		} else {

			khp = padding+(i+1) + (j+padding)*n_pad;
			khm = padding+(i-1) + (j+padding)*n_pad;

			hx = (res[khp] - res[khm])*(s_paras.inv_dxt2);
			hxdbhtl = hx * inv_res*lambda_x;

			//== Approximating (hx/h)^{n+1} (time, n+1)
			hx_nxt = (res_nxt[khp] - res_nxt[khm])*(s_paras.inv_dxt2);
			hxdbhtl += hx_nxt * inv_res_nxt * lambda_x;
			hxdbhtl *= 0.5;

			vecy[ind_start+j]=(mu_x-hxdbhtl)*res_temp[km]+(1.0-2.0*mu_x)*res_temp[k]+(mu_x+hxdbhtl)*res_temp[kp]+ 
							Qterm_stored[ind_start + j];

		}

		if(j==0){
			vecy[ind_start+0] += hx0*m_paras.E_S;	//== South boundary y=0
		} else if(j==(m-1)){
			vecy[ind_start+m-1] += hxm*m_paras.E_N;	//== North boundary y=L
		}

		int khc = padding+i + (j+padding)*n_pad;
		int kc = j + i*m;

		
		if(is_predict){
			vecy[ind_start+j] += 0.5*dt_temp_evl*(C*m_paras.K1_t_kratio_db2 / (res[khc])*T_int[kc]);
		} else{
			vecy[ind_start+j] += 0.5*dt_temp_evl*((C*m_paras.K1_t_kratio_db2 / (res[khc])*T_int[kc]) +
			D*m_paras.K1_t_kratio_db2 / (res_nxt[khc])*T_int_corrector[kc]);
		}

	
}

__global__ void transpose_arrays(double* A, double* A_trans, double* B, double* B_trans, double* C, double* C_trans, int n, int m, int n_pad, int m_pad, int padding)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k_A, k_B;
	
	if (idx < n && idy < m)
	{
		k_A = idy + m*idx;
		k_B = padding+idy + (idx+padding)*m_pad;
		A_trans[k_A] = A[idx+idy*n];
		B_trans[k_B] = B[padding+idx + (idy+padding)*n_pad];
		C_trans[k_B] = C[padding+idx + (idy+padding)*n_pad];
	}
}

__global__ void reverse_transpose(double* A_trans, double* A, int n, int m)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k = idx + idy * n;

	if (idx < n && idy < m)
	{
		A[k] = A_trans[idy + m * idx];
	}
}

__device__ void get_mat_Crank_z(double dt_temp_evl, double* mat_sub, double* mat_main, double* mat_sup, double mu_z, int p, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, int ind_start
, double* muzA, double* muzB, double* muzC, double Bi_new)
{
	
	mat_sub[ind_start+0] = 0;
	mat_main[ind_start+0] = 1 - muzB[0];							//Note no flux boundary condition at x=0; T_{-1,j}=T_{0,j}
	mat_sup[ind_start+0] = -muzC[0];

    for(int k=1; k <= (p-3); k++){						
													//Check the index length for x-direction here: Is it n?
        mat_sub[ind_start+k] = -muzA[k];
        mat_main[ind_start+k] = 1 - muzB[k];
        mat_sup[ind_start+k] = -muzC[k];
	}

	mat_sub[ind_start+p - 2] = -muzA[p-2];
	mat_main[ind_start+p - 2] = 1 - muzB[p-2] - muzC[p-2] * (1.0 - Bi_new) / (1.0 + Bi_new);
    mat_sup[ind_start+p - 2] = 0;
    
}

__device__ void get_vec_Crank_z(double dt_temp_evl, double* Tsub, int idx, int idy, double vecz[], double mu_z, int n, int m, int p, double* res_temp, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, int ind_start
, double* muzA, double* muzB, double* muzC, double Bi_new)
{
    double T_ghost;

    int ind_c = idy*n*p + idx*p;	// Centered index at z=0; Now in block with beginning index idy*n*p+idx*p+0
	int ind_p = ind_c + 1;		// Index at z=1;
	int ind_pp = ind_p + 1;		// Index at z=2;
	

	//Input Tsub as a vector of length Nzs-1
	vecz[ind_start+0]= muzA[0]*( Tsub[ind_c] + res_temp[idx+n*idy] ) + ( 1 + muzB[0]) * Tsub[ind_p] + muzC[0] * Tsub[ind_pp];

	for(int k=1;k <= p-3;k++){
        ind_c = idy*n*p + idx*p + k;
		ind_p = ind_c + 1;
		ind_pp = ind_p + 1;
		vecz[ind_start+k]=muzA[k]*Tsub[ind_c]+(1 + muzB[k])*Tsub[ind_p]+muzC[k]*Tsub[ind_pp];
	}

    ind_c = idy*n*p + idx*p + p-1;
	
	T_ghost = 0.5*(1.0 - Bi_new) / (1 + Bi_new)*Tsub[ind_c] + (2.0*Bi_new) / (1.0 + Bi_new)*m_paras.Temp_amb;		// Define ghost point for Newton Law of Cooling

    ind_c = idy*n*p + idx*p + p-2;
	ind_p = ind_c + 1;

	vecz[ind_start+p - 2] = muzA[p-2] * Tsub[ind_c] + (1 + muzB[p-2])*Tsub[ind_p] + 2.0*muzC[p-2]*T_ghost;

}

void update_sub_temp(double dt_temp_evl, double* res_temp, double* Tsub, double mu_z, int n, int m, int p, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras
, double* dev_muzA, double* dev_muzB, double* dev_muzC, double* dev_muzD, double Bi_new, double* Tsub_int1, double* Tsub_int2, double mu_x, double mu_y
, double* Tsub_ikj, double* Tsub_jki, double* Tsub_int1_jki, double* Tsub_int2_kij, double* Az_ijk_stored, double* Az_kij_stored, double* Ay_jki_stored, double* Ay_ijk_stored)
{
	double D_term, E_term;
	
	//== Define Number of threads and blocks
	//-- Note: Fix number of threads to 32*32 = 1024 threads and make blocks to fill the rest.
	dim3 threadsPerBlock(min((int) tPBx, n), min( (int) tPBy, m));
	dim3 numBlocks((n+threadsPerBlock.x-1)/threadsPerBlock.x , (m+threadsPerBlock.y-1)/threadsPerBlock.y );

	if(m_paras.nonlin_sub_heat)
	{
		//== Nonlinear equation, use Newton's method with Crank-Nicolson Scheme
		D_term = (1.0-Bi_new)/(1.0+Bi_new);
		E_term = (2.0*Bi_new)/(1.0+Bi_new)*m_paras.Temp_amb;	
		Newton_Crank_z<<<numBlocks,threadsPerBlock>>>(dt_temp_evl, Tsub, res_temp, n, m, p, dev_muzA, dev_muzB, dev_muzC, dev_muzD, Bi_new, m_paras, D_term, E_term);

	} else {
		if(!m_paras.sub_in_plane_diff)
		{
			//== In-plane diffusion neglected. Equation is T_t = K_2 T_{zz}
			//== Linear equation, use Crank-Nicolson scheme
			//== Solve for Tsub at time t_{n+1}
			Crank_z<<<numBlocks,threadsPerBlock>>>(dt_temp_evl, res_temp, Tsub, mu_z, n, m, p, m_paras, s_paras, dev_muzA, dev_muzB, dev_muzC, Bi_new);
		} else{

			dim3 nt_store(min(4, (int) s_paras.n), min(4, (int) s_paras.m), min(9, (int) (s_paras.p-1)) );
			dim3 nB_store((s_paras.n+nt_store.x-1)/nt_store.x, (s_paras.m+nt_store.y-1)/nt_store.y,(s_paras.p-1+nt_store.z-1)/nt_store.z);

			D_term = (1.0-Bi_new)/(1.0+Bi_new);
			E_term = (2.0*Bi_new)/(1.0+Bi_new)*m_paras.Temp_amb;

			transpose_sub_1<<<nB_store,nt_store>>>(Tsub, Tsub_ikj, Tsub_jki, n, m, p);

			dim3 adiz1_tpb(THREADBLOCK_SIZE_X,1,1);

			//== Define number of blocks. For y-direction 8 is optimal for 181x8 size domain.
			dim3 adiz1_blocks(1,(m+adiz1_tpb.y-1)/adiz1_tpb.y,(p-1+adiz1_tpb.z-1)/adiz1_tpb.z);

			ADIZ1<<<adiz1_blocks, adiz1_tpb>>>(Tsub, Tsub_int1, res_temp, n, m, p, mu_x, mu_y, m_paras.Temp_amb, Bi_new, dev_muzA, dev_muzB, dev_muzC, Tsub_ikj, Tsub_jki, Az_ijk_stored, Ay_jki_stored, D_term, E_term, m_paras);
			cudaDeviceSynchronize();

			transpose_A_mats<<<nB_store, nt_store>>>(Az_ijk_stored, Az_kij_stored, Tsub_int1, Tsub_int1_jki, n, m, p);

			dim3 adiz2_tpb(1,THREADBLOCK_SIZE_Y,1);
			dim3 adiz2_blocks((n+adiz2_tpb.x-1)/adiz2_tpb.x,1,(p-1+adiz2_tpb.z-1)/adiz2_tpb.z);
			ADIZ2<<<adiz2_blocks, adiz2_tpb>>>(Tsub_int1, Tsub_int2, Tsub_jki, res_temp, n, m, p, mu_y, Ay_jki_stored, m_paras);

			dim3 adiz3_tpb(1,1);
			dim3 adiz3_blocks((n+adiz3_tpb.x-1)/adiz3_tpb.x,(m+adiz3_tpb.y-1)/adiz3_tpb.y);
			
			ADIZ3<<<numBlocks, threadsPerBlock>>>(Tsub_int2, Tsub, res_temp, n, m, p, dev_muzA, dev_muzB, dev_muzC, m_paras.Temp_amb, Bi_new, Az_kij_stored, D_term, E_term);

		}

	}

	//================================================//
}

__global__ void Crank_z(double dt_temp_evl, double* res_temp, double* Tsub, double mu_z, int n, int m, int p, model_parameters<double> m_paras, spatial_parameters<double> s_paras 
, double* muzA, double* muzB, double* muzC, double Bi_new)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int ind_c;
	int ind_start;

	__shared__ double vecz[(tPBz)*tPBx*tPBy];
	__shared__ double matz_sub[(tPBz)*tPBx*tPBy];
	__shared__ double matz_main[(tPBz)*tPBx*tPBy];
	__shared__ double matz_sup[(tPBz)*tPBx*tPBy];

    if (idx < n && idy < m)
    {
		while(idy < m){
			while(idx < n){

			
				ind_start = (p-1) * (threadIdx.x + (tPBx) * threadIdx.y);
				//== Build matrices on device (Note that all the matrices are identical for each thread/block)
				get_mat_Crank_z(dt_temp_evl, matz_sub, matz_main, matz_sup, mu_z, p, m_paras, s_paras, ind_start, muzA, muzB, muzC, Bi_new);

				//== Build device vectors for given (x,y) --> (idx,idy)
				get_vec_Crank_z(dt_temp_evl, Tsub, idx, idy, vecz, mu_z, n, m, p, res_temp, m_paras, s_paras, ind_start, muzA, muzB, muzC, Bi_new);

				//== Solve Ax=b, where A is matz and b is vecz
				tri_solve(matz_sub, matz_main, matz_sup, vecz, p-1, ind_start, ind_start);
				//__syncthreads();
				//--Note: about 4ms to run tri_solve

				//== Update Tsub at the appropriate indices
				for (int k=0; k<(p-1); k++)
				{
					//ind_c = k+idy*p+idx*m*p;
					ind_c = k+idx*p+idy*n*p;        // Changed 3/17/20
					// ind_c = k + ind_c_right;
					Tsub[ind_c+1] = vecz[ind_start+k];
				}
				// __syncthreads();

				// Move on to new indices
				idx += blockDim.x * gridDim.x;
			}
		// Reset idx
			idx = threadIdx.x + blockIdx.x * blockDim.x;
			idy += blockDim.y * gridDim.y;
		}
	}
}

__global__ void update_substrate_interface(double* Tsub, double* T_int, double inv_dzst2, int n, int m, int p, double* T_int_coef, model_parameters<double> m_paras)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
	//== Indices
	int k = idy + m*idx;
    int sub_kc = 0 + p*idx + n*p*idy;
    int sub_kp = sub_kc + 1;
    int sub_kpp = sub_kp + 1;

	double coef1, coef2, coef3;
	double Tsub1, Tsub2, Tsub3;
	double TC, dTC, ddTC;
	
    if(idx < n && idy < m)
    {
		coef1 = T_int_coef[0];
		coef2 = T_int_coef[1];
		coef3 = T_int_coef[2];
		Tsub1 = Tsub[sub_kc];
		Tsub2 = Tsub[sub_kp];
		Tsub3 = Tsub[sub_kpp];

		//== Uniform grid here
		// T_int[k] = (3*Tsub[sub_kc]-4*Tsub[sub_kp]+Tsub[sub_kpp])*(inv_dzst2);

		//== Non-uniform grid here
		// T_int[k] = T_int_coef[0]*Tsub[sub_kc]+T_int_coef[1]*Tsub[sub_kp]+T_int_coef[2]*Tsub[sub_kpp];

		// if(m_paras.nonlin_sub_heat){
		get_TC_funcs(Tsub[sub_kc], &TC, &dTC, &ddTC, m_paras.T_scl, m_paras.TC_model);

		// T_int(i)=(T_int_coef(1)*res_temp(i,n_film+1)+T_int_coef(2)*res_temp(i,n_film+2)+T_int_coef(3)*res_temp(i,n_film+3))*TC
		T_int[k] = (coef1 * Tsub1 + coef2 * Tsub2 + coef3 * Tsub3) * TC;
	}


}

__global__ void store_vars_at_prev_step(double* res_temp, double* res_temp_prev, double* res_temp_trans, double* res_temp_trans_prev, double* T_int, double* T_int_prev, double* Tsub, double* Tsub_prev, int n, int m, int p)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int k = idx + n*idy;
    int k_B = idy + m*idx;
	int kz;
    
    if(idx < n && idy < m)
    {
        res_temp_prev[k] = res_temp[k];
        T_int_prev[k_B] = T_int[k_B];
        res_temp_trans_prev[k_B] = res_temp_trans[k_B];

    }

}

__global__ void store_Tsub_at_prev_step(double* Tsub, double* Tsub_prev, int n, int m, int p)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	int kz;

	double A;
	

	if(idx < n && idy < m && idz < p)
	{
		kz = idz + idx*p + idy*n*p;
		A = Tsub[kz];
		// Tsub_prev[kz] = Tsub[kz];
		Tsub_prev[kz] = A;
	}
}

__global__ void compute_err_local(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp, double* res_temp_trans, double* res_temp_prev
    , double* res_temp_trans_prev, double* T_int_prev, double* T_int, double* err_local, model_parameters<double> m_paras, spatial_parameters<double> s_paras, int n, int m, int n_pad, int m_pad, int padding
	, double F_of_t, double F_of_t_prev)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int k = idx + idy * n;

	if(idx < n & idy < m)
	{
		err_local[k] = abs(res_temp[k] - res_temp_prev[k]) + dt_temp_evl 
			* abs(RHS_func(idx, idy, res_temp, res_temp_trans, res_nxt, res_nxt_trans, T_int, t, m_paras, s_paras, n, m, n_pad, m_pad, padding, F_of_t) 
			- RHS_func(idx, idy, res_temp_prev, res_temp_trans_prev, res, res_trans, T_int_prev, t - dt_temp_evl, m_paras, s_paras, n, m, n_pad, m_pad, padding, F_of_t_prev));
	}
}	

__device__ double RHS_func(int i, int j, double* T_in, double* T_in_trans, double* h_in, double* h_in_trans, double* T_int_in, double time_in
    , model_parameters<double> &m_paras, spatial_parameters<double> &s_paras, int n, int m, int n_pad, int m_pad, int padding, double F_of_t)
{
    double hx, hy, Tx, Ty;
	double Txx, Tyy;
	double A, Bx, By, C;
    double RHS;
    int kc, kp, km, khc, khp, khm;
    int khc_trans, khp_trans, khm_trans;
    int kc_trans, kp_trans, km_trans;

    //== Define base indices that don't change in the conditional statements
    khc = padding+i + (j+padding)*n_pad;
    khc_trans = padding+j + (i+padding)*m_pad;
    kc = i + n * j;
    kc_trans = j + m * i;

	if (i == 0) {
        khp = khc + 1;
        kp = kc + 1;
		hx = (h_in[khp] - h_in[khc])*s_paras.inv_dxt2;
		Tx = (T_in[kp] - T_in[kc])*s_paras.inv_dxt2;
		Txx = (T_in[kp] - T_in[kc])*s_paras.inv_dx2;
	}
	else if (i==(n-1)){
        khm = khc - 1;
        km = kc - 1;
		hx = (h_in[khc] - h_in[khm])*s_paras.inv_dxt2;
		Tx = (T_in[kc] - T_in[km])*s_paras.inv_dxt2;
		Txx = (T_in[km] - T_in[kc])*s_paras.inv_dx2;
	}
	else {
        kp = kc + 1;
        km = kc - 1;
        khm = khc - 1;
        khp = khc + 1;

		hx = (h_in[khp] - h_in[khm])*s_paras.inv_dxt2;
		Tx = (T_in[kp] - T_in[km])*s_paras.inv_dxt2;
		Txx = (T_in[kp] - 2.0*T_in[kc] + T_in[km])*s_paras.inv_dx2;
	}

	if (j == 0) {
        khp_trans = khc_trans + 1;
        kp_trans = kc_trans + 1;
		hy = (h_in_trans[khp_trans] - h_in_trans[khc_trans])*s_paras.inv_dyt2;
		Ty = (T_in_trans[kp_trans] - T_in_trans[kc_trans])*s_paras.inv_dyt2;
		Tyy = (T_in_trans[kp_trans] - T_in_trans[kc_trans])*s_paras.inv_dy2;
	}
	else if (j == (m-1)) {
        khm_trans = khc_trans - 1;
        km_trans = kc_trans - 1;
		hy = (h_in_trans[khc_trans] - h_in_trans[khm_trans])*s_paras.inv_dyt2;
		Ty = (T_in_trans[kc_trans] - T_in_trans[km_trans])*s_paras.inv_dyt2;
		Tyy = (T_in_trans[km_trans] - T_in_trans[kc_trans])*s_paras.inv_dy2;
	}
	else {
        khp_trans = khc_trans + 1;
        khm_trans = khc_trans - 1;
        kp_trans = kc_trans + 1;
        km_trans = kc_trans - 1;
		hy = (h_in_trans[khp_trans] - h_in_trans[khm_trans])*s_paras.inv_dyt2;
		Ty = (T_in_trans[kp_trans] - T_in_trans[km_trans])*s_paras.inv_dyt2;
		Tyy = (T_in_trans[kp_trans] - 2.0*T_in_trans[kc_trans] + T_in_trans[km_trans])*s_paras.inv_dy2;
	}

	double inv_res = 1.0 / h_in[khc];
	
		
	A = m_paras.K1;
	Bx = m_paras.K1*hx * inv_res;
	By = m_paras.K1*hy/(h_in_trans[khc_trans]);
	C = -1.0*m_paras.K1*m_paras.k_ratio * inv_res;
	RHS = A * (Txx + Tyy) + Bx * Tx + By * Ty + C * T_int_in[kc_trans] + Q_bar(h_in[khc], time_in, m_paras, inv_res, F_of_t);

	return RHS;
}

void check_local_trunc_err(double t, double dt_temp_evl, double* res, double* res_nxt, double* res_trans, double* res_nxt_trans, double* res_temp_prev, double* res_temp, double* res_temp_trans, double* res_temp_trans_prev
    , double* T_int_prev, double* T_int, int &temp_dbg, double avg_temp, model_parameters<double> &m_paras, spatial_parameters<double> &s_paras
 , int n, int m, int n_pad, int m_pad, int padding, double* err_local, double* h_max, double* d_max, int* d_mutex, double F_of_t, double F_of_t_prev)
{
	
    double err_trunc = 0.0;
    double rel_err; //, host_err_local[n*m];
    dim3 threadsPerBlock(min(n,32),min(m,32));
    dim3 numBlocks((n+threadsPerBlock.x-1)/threadsPerBlock.x, (m+threadsPerBlock.y-1)/threadsPerBlock.y);
    int k;

	dim3 gridSize = 64;
	dim3 blockSize = 64;

    //== First determine the local error at each point on the device; Save error into err_local[k], where k = i + n*j
    compute_err_local<<<numBlocks, threadsPerBlock>>>(t, dt_temp_evl, res, res_nxt, res_trans, res_nxt_trans, res_temp, res_temp_trans, res_temp_prev
        , res_temp_trans_prev, T_int_prev, T_int, err_local, m_paras, s_paras, n, m, n_pad, m_pad, padding, F_of_t, F_of_t_prev);
	cudaDeviceSynchronize();

	//== Find maximum using reduction with shared memory
	//== See: https://bitbucket.org/jsandham/algorithms_in_cuda/src/master/find_maximum/
	find_maximum_kernel<<< gridSize, blockSize >>>(err_local, d_max, d_mutex, n*m);
	cudaMemcpy(h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);

	err_trunc = *h_max;

	//== Reset h_max and d_max so maximum doesn't get stuck at previous value
	*h_max = -1;
	cudaMemcpy(d_max, h_max, sizeof(double), cudaMemcpyHostToDevice);


    //== Define the relative error to be the maximum truncation error normalized by the average temperature
	rel_err = err_trunc / avg_temp;

	//== Check if truncation error is large
	if (rel_err >= m_paras.temp_tol) {
		temp_dbg = 1;
	}
	else {
		temp_dbg = 0;
	} 
}

__global__ void set_film_sub_interface_temp(double* res_temp, double* Tsub, int n, int m, int p)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k = idx + idy * n;
	int ind_zero = k*p; //Index at z=0:  0+idx*p+idy*n*p

	if(idx < n && idy < m)
	{
		Tsub[ind_zero] = res_temp[k];
	}

}

__global__ void revert_vars_to_prev_val(double* res_temp, double* res_temp_prev, double* res_temp_trans, double* res_temp_trans_prev, double* T_int, double* T_int_prev, double* Tsub, double* Tsub_prev, int n, int m, int p)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int k = idx + n*idy;
    int k_B = idy + m*idx;
	int kz;
    
    if(idx < n && idy < m)
    {
        res_temp[k] = res_temp_prev[k];
        T_int[k_B] = T_int_prev[k_B];
        res_temp_trans[k_B] = res_temp_trans_prev[k_B];

    }
}

__global__ void revert_Tsub_to_prev_val(double* Tsub, double* Tsub_prev, int n, int m, int p)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	int kz;

	if(idx < n && idy < m && idz < p)
	{
		kz = idz + idx*p + idy*n*p;
		Tsub[kz] = Tsub_prev[kz];
	}
}

__global__ void find_maximum_kernel(double *array, double *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ double cache[256];


	double temp = -1.0;
	while(index + offset < n){
		temp = fmax(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmax(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmax(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

__global__ void sum_reduction( double *a, double *c, int a_len ) {
	//== Code taken from "Cuda by Example" by Sanders & Kandrot
	//== Used dot product (chapter 5) code but second vector is 1
    __shared__ double cache[TPB];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double   temp = 0;
    while (tid < a_len) {
        temp += a[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


__global__ void update_res_temp(double* vecx, double* res_temp, int n, int m)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k;

	if (idx < n && idy < m)
	{
		k = idx + idy * n;
		res_temp[k] = vecx[k];
	}
}

__global__ void update_res_temp_trans(double* vecy, double* res_temp_trans, int n, int m)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int k;

	if (idx < n && idy < m)
	{
		k = idy + idx * m;
		res_temp_trans[k] = vecy[k];
	}
}

__device__ void get_TC_funcs(double T, double* TC, double* dTC, double* ddTC, double Temp_scl, int TC_model)
{
	//==Description: Determine temperature-varying thermal conductivity
	// Model 1: constant k=1
	// Model 2: Cubic polynomial
	// Model 3: Cubic polynomial with sigmoids
	//== Register variables
	double Temp_scl_fit = 45.086711287412051;			//== Temperature fit was done in
	double T_fit = T * Temp_scl / Temp_scl_fit;			//== Temperature to use in fit
	double beta = 40;									//== Temperature shift for sigmoid function
	double k_soften = 2.0 / 1.4;						//== Thermal Conductivity at SiO2 softening temperature
	double a,b,c,d;										//== Cubic fitting parameters
	double p3, p3p, p3pp, f1, f1p, f1pp, f2, f2p, f2pp;

	double expterm;

	switch(TC_model) {
		case 1 :
			*TC = 1.0;
			*dTC = 0.0;
			*ddTC = 0.0;
			break;

		case 2 :
			a = -0.0001228;										//== Coefficient of T^3
			b = 0.006845; 										//== Coefficient of T^2
			c = -0.06601;										//== Coefficient of T
			d = 1.178;											//== Coefficient of 1
			p3 = d + c*(T_fit) + b*T_fit*T_fit + a*T_fit*T_fit*T_fit;
			p3p = 3.0 * a * T_fit*T_fit + 2.0*b*T_fit + c;
			p3pp = 6.0 * a * T_fit + 2.0 * b;
			*TC = p3;
			*dTC = p3p;
			*ddTC = p3pp;
			break;

		case 3:
			a = -0.0001228;										//== Coefficient of T^3
			b = 0.006845; 										//== Coefficient of T^2
			c = -0.06601;										//== Coefficient of T
			d = 1.178;											//== Coefficient of 1
			p3 = d + c*(T_fit) + b*T_fit*T_fit + a*T_fit*T_fit*T_fit;
			p3p = 3.0 * a * T_fit*T_fit + 2.0*b*T_fit + c;
			p3pp = 6.0 * a * T_fit + 2.0 * b;

			expterm = exp(T_fit-beta);
			f1 = 1.0 / (1.0+expterm);

			// f1 = 1.0 / (1.0+exp(T_fit - beta));
			f1p = (f1-1.0) * f1;
			// f1pp = (2.0 * f1 - 1) * (f1-1) * f1;
			f1pp = (2.0 * f1 - 1) * f1p;//(f1-1) * f1;

			f2 = f1*expterm;
			// f2=1.0 / (1.0+exp(beta-T_fit));
			f2p = f2 * (1.0-f2);
			f2pp = (2.0 * f2 - 1) * (f2-1) * f2;
			// f2pp = (1.0-2.0 * f2) * (f2p); // (f2-1) * f2;

			*TC = f1 * p3 + f2 * k_soften;
			*dTC = f1 * p3p + f1p * p3 + f2p * k_soften;
			*ddTC = f1 * p3pp + 2.0 * f1p * p3p + f1pp * p3p + f2pp * k_soften;
			break;
	}
	
}

__global__ void Newton_Crank_z(double dt_temp_evl, double* Tsub, double* res_temp, int n, int m, int p, double* dz_A, double* dz_B, double* dz_C, double* dz_D, double Bi_new, model_parameters<double> m_paras
, double D_bdry_term, double E_bdry_term)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int ind_c;
	int ind_start;

	__shared__ double b_total[(tPBz)*tPBx*tPBy];
	__shared__ double Jzm[(tPBz)*tPBx*tPBy];
	__shared__ double Jz0[(tPBz)*tPBx*tPBy];
	__shared__ double Jzp[(tPBz)*tPBx*tPBy];

	__shared__ double func_guess[(tPBz)*tPBx*tPBy];
	__shared__ double b_fix[(tPBz)*tPBx*tPBy];

	__shared__ double A[(tPBz)*tPBx*tPBy];
	__shared__ double B[(tPBz)*tPBx*tPBy];
	__shared__ double C[(tPBz)*tPBx*tPBy];
	__shared__ double D[(tPBz)*tPBx*tPBy];
	__shared__ double dA[(tPBz)*tPBx*tPBy];
	__shared__ double dB[(tPBz)*tPBx*tPBy];
	__shared__ double dC[(tPBz)*tPBx*tPBy];
	__shared__ double dD[(tPBz)*tPBx*tPBy];
	__shared__ double T_guess[(tPBz)*tPBx*tPBy];

	int iflag_tri, T_newton_iter;
	double T_top_next, T_top_fix;
	double T_err_newton, T_err_posit, T_err_resid, T_err_local, T_err_trunc, small;
	
	small=1.0e-20;
	T_newton_iter = 0;
	T_err_newton = 1.0;


	if (idx < n && idy < m)
    {
		while(idy < m){
			while(idx < n){

				//== Solver code goes here
				//==First set the starting index
				ind_start = (p-1) * (threadIdx.x + (tPBx) * threadIdx.y);
				ind_c = idx*p+idy*n*p;

				//== First set the fixed portion of temperature
				//== Note we use T_guess to initially store T_fix as T_guess=T_fix initially anyways.
				//== This saves storage.
				for(int k=0; k<p-1; k++){
					T_guess[ind_start+k] = Tsub[ind_c+k+1];
				}

				//== Set the coefficients of spatial discretization
				get_thermal_coef(T_guess, A, B, C, D, dA, dB, dC, dD, p, m_paras.T_scl, dz_A, dz_B, dz_C, dz_D, Bi_new, ind_start, m_paras.TC_model);

				//== Get fixed component of function that is not iterated using T_fix (temperature at time t^n)
				T_top_fix = Tsub[ind_c];

				//== Store fixed function f(t) into func_guess. This component remains fixed for Newton iteration
				get_Newton_func(func_guess, T_guess, T_top_fix, A, B, C, D, Bi_new, p, ind_start, D_bdry_term, E_bdry_term, m_paras.Temp_amb, m_paras.T_scl, m_paras.TC_model);

				for(int k=0; k<p-1; k++){
					//== Compute fixed RHS. This stays constant during iteration
					b_fix[ind_start+k] = T_guess[ind_start+k] + 0.5 * dt_temp_evl * func_guess[ind_start+k];
				}

				//== Begin Newton iteration for determining guess/correction combo
				while(T_err_newton>m_paras.T_eps_newt)
				{
					for(int k=0;k<p-1;k++){
						//== Begin by computing RHS at each iteration. This starts at 0.
						b_total[ind_start+k] = -T_guess[ind_start+k] + 0.5 * dt_temp_evl * func_guess[ind_start+k] + b_fix[ind_start+k];						
					}


					//== Build Jacobian
					T_top_next = res_temp[idx+n*idy];
					// printf("T_top_next = %f\n", T_top_next);

					get_Jacobian(Jzm, Jz0, Jzp, T_guess, T_top_next, A, B, C, D, dA, dB, dC, dD, p, Bi_new, ind_start, D_bdry_term, E_bdry_term, m_paras.Temp_amb, m_paras.T_scl, m_paras.TC_model);
					
					//== Build LHS matrix
					get_Newton_LHS(dt_temp_evl, Jzm, Jz0, Jzp, p, ind_start);

					//== Check to make sure the jacobian does not contain NaN's before solving.
					for(int k=0;k<p-1;k++)
					{
						if( isnan(Jzm[ind_start+k]) || isnan(Jz0[ind_start+k]) || isnan(Jzp[ind_start+k]) )
						{
							printf("k = %d\n", k);
							printf("sub-diag input matrix NaN\n");
							exit;
						}
					}

					//== Solve the tridiagonal system Jx=F
					//== Note: Solution stored in b
					tri_solve(Jzm, Jz0, Jzp, b_total, p-1, ind_start, ind_start);

					//== Usually a iflag_tri is exited. In this case we didn't use that.
					//== Check to make sure this doesn't make an error.

					T_err_newton=0.0;
					T_err_posit=1.0e2;

					for(int k=0;k<p-1;k++)
					{
						//== Check if Newton's method converged
						if(T_guess[ind_start+k]<(1.0e-15)){
							T_err_resid=abs(b_total[ind_start+k]/(1.0e-15));
						} else{
							T_err_resid=abs(b_total[ind_start+k]/T_guess[ind_start+k])	;
						}
			
						if(T_err_resid>T_err_newton)
						{
							T_err_newton=T_err_resid;	
						}
						
						//== Update the guess and check for positivity
						//== Update iteration T_guess = T* + correction
						T_guess[ind_start+k] = T_guess[ind_start+k]+b_total[ind_start+k];
						
						if(T_guess[ind_start+k]<T_err_posit){
							T_err_posit = T_guess[ind_start+k];
						}
						
					}

					if(T_err_posit < small){
						printf("Negative solution, dt= %f\n", dt_temp_evl);
						printf("            err_posit= %f\n", T_err_posit);
					}

					//== Iterative solution updated. Recompute coefficients and recompute func_guess
					get_thermal_coef(T_guess, A, B, C, D, dA, dB, dC, dD, p, m_paras.T_scl, dz_A, dz_B, dz_C, dz_D, Bi_new, ind_start, m_paras.TC_model);
					get_Newton_func(func_guess, T_guess, T_top_next, A, B, C, D, Bi_new, p, ind_start, D_bdry_term, E_bdry_term, m_paras.Temp_amb, m_paras.T_scl, m_paras.TC_model);

					T_newton_iter += 1;
					if(T_newton_iter > m_paras.n_iter_max){
						exit;
					}

				}

				//== Commit solution to global memory
				for(int k=0;k<p-1;k++){
					Tsub[ind_c+k+1] = T_guess[ind_start+k];
				}

				//== Incrementing idx to cover entire grid
				idx += blockDim.x * gridDim.x;
			}
		// Reset idx
			idx = threadIdx.x + blockIdx.x * blockDim.x;
			idy += blockDim.y * gridDim.y;
		}
	}

}

__device__ void get_Jacobian(double* Jzm, double* Jz0, double* Jzp, double* T_guess, double T_top, double* A, double* B, double* C, double* D,
		double* dA, double* dB, double* dC, double* dD, int p, double Bi_new, int ind_start, double D_term, double E_term, double Temp_amb, double Temp_scl, int TC_model)
{
	double T_ghost;
	double Tz0;
	
	//== Second, note T_top represents the node T_0 which we have here as T_guess(0)
	Tz0 = T_guess[ind_start+1] - T_top;
	Jzm[ind_start+0] = 0.0;
	Jz0[ind_start+0] = dA[ind_start+0] * T_top + B[ind_start+0] + dB[ind_start+0] * T_guess[ind_start+0] + dC[ind_start+0] * T_guess[ind_start+1] +
			dD[ind_start+0] * Tz0 * Tz0;
	Jzp[ind_start+0] =  C[ind_start+0] + 2.0 * D[ind_start+0] * Tz0;

    for(int k=1;k<p-2; k++)
	{
		Tz0 = T_guess[ind_start+k-1] - T_guess[ind_start+k+1];
        Jzm[ind_start+k] = A[ind_start+k] + D[ind_start+k] * 2.0 * Tz0;
		Jz0[ind_start+k] = dA[ind_start+k] * T_guess[ind_start+k-1] + B[ind_start+k] + dB[ind_start+k] * T_guess[ind_start+k] + dC[ind_start+k] * T_guess[ind_start+k+1]
				+ dD[ind_start+k] * Tz0 * Tz0;
        Jzp[ind_start+k] = C[ind_start+k] - D[ind_start+k] * 2.0 * Tz0;
	}

	//== Get ghost point by solving nonlinear equation ks*dT/dz = Bi(Ts-Ta);
	sub_newton(T_ghost, T_guess[p-2], Bi_new, Temp_amb, Temp_scl, TC_model);

	Tz0 = T_guess[ind_start+p-3] - T_ghost;
    Jzm[ind_start+p-2] = A[ind_start+p-2] + D[ind_start+p-2] * 2.0 * Tz0;
	Jz0[ind_start+p-2] = dA[ind_start+p-2] * T_guess[ind_start+p-3] + B[ind_start+p-2] + dB[ind_start+p-2] * T_guess[ind_start+p-2]
					+ dC[ind_start+p-2] * T_ghost + C[ind_start+p-2]*D_term + dD[ind_start+p-2] * Tz0 * Tz0;
	Jzp[ind_start+p-2] = 0.0;

}


__device__ void get_Newton_LHS(double dt, double* Jzm, double* Jz0, double* Jzp, int p, int ind_start)
{
	//== Get matrix for Newton's method (del_ij - 0.5dt*F_ij), for jacobian F_ij (Jzm,Jz0,Jzp) 
    for(int k=0;k<p-1;k++)
	{
		Jzm[ind_start+k] = -0.5 * dt * Jzm[ind_start+k];
        Jz0[ind_start+k] = 1.0 - 0.5 * dt * Jz0[ind_start+k];
		Jzp[ind_start+k] = -0.5 * dt * Jzp[ind_start+k];
	}
}

__device__ void get_thermal_coef(double* Temp_in, double* A, double* B, double* C, double* D, double* dA, double* dB, double* dC, double* dD, int p, double Temp_scl
, double* dz_A, double* dz_B, double* dz_C, double* dz_D, double Bi_new, int ind_start, int TC_model)
{
	double TC, dTC, ddTC;

	for(int k=0; k<p-1; k++)
	{

		get_TC_funcs(Temp_in[ind_start+k], &TC, &dTC, &ddTC, Temp_scl, TC_model);

		//== A,B,C coefficients for approximation of d/dz( k(T)dT/dz ) on nonuniform grid
		//== Approximation does backwards diff. inside and forward outside
		A[ind_start+k] = dz_A[k] * TC;										//==Coefficient of T_{i-1}
		B[ind_start+k] = dz_B[k] * TC;										//==Coefficient of T_{i}
		C[ind_start+k] = dz_C[k] * TC;										//==Coefficient of T_{i+1}
		D[ind_start+k] = dz_D[k] * dTC;										//==Coefficient of T_{i+1}


		//== Derivatives of A,B,C w.r.t temperature
		dA[ind_start+k] = dz_A[k] * dTC;									//==Derivative of A w.r.t. T_{i}
		dB[ind_start+k] = dz_B[k] * dTC;									//==Derivative of B1 w.r.t. T_{i}
		dC[ind_start+k] = dz_C[k] * dTC;									//==Derivative of C w.r.t. T_{i+1}
		dD[ind_start+k] = dz_D[k] * ddTC;									//==Second Derivative of D w.r.t T_{i}
		// dD[ind_start+k] = dz_D[ind_start+k] * ddTC_func(Temp_in[ind_start+k]);			//==Second Derivative of D w.r.t T_{i}
	}
}

__device__ void get_Newton_func(double* func, double* T_in, double T_top, double* A, double* B, double* C, double* D, double Bi_new, int p, int ind_start, double D_term, double E_term
, double Temp_amb, double Temp_scl, int TC_model)
{
	double T_ghost;
	double Tz0;


	//== Function F(T_{i-1},T_i, T_{i+1}) = A*T_{i-1} + B*T_{i} + C*T_{i+1}, with
	//==== variable coefficients A,B,C from thermal conductivity temperature dependence

	//== Index j=1
	Tz0 = T_top - T_in[ind_start+1];
	func[ind_start+0]= A[ind_start+0] * T_top + B[ind_start+0] * T_in[ind_start+0] + C[ind_start+0] * T_in[ind_start+1]
			+ D[ind_start+0] * Tz0 * Tz0;

    for(int j=1;j<p-2;j++)
	{
		Tz0 = T_in[ind_start+j-1] - T_in[ind_start+j+1];
		func[ind_start+j] = A[ind_start+j] * T_in[ind_start+j-1] + (B[ind_start+j]) * T_in[ind_start+j] + C[ind_start+j] * T_in[ind_start+j+1]
				 + D[ind_start+j] * Tz0 * Tz0;
	}

	//== Get ghost point by solving nonlinear equation ks*dT/dz = Bi(Ts-Ta);
	sub_newton(T_ghost, T_in[ind_start+p-2], Bi_new, Temp_amb, Temp_scl, TC_model);

	Tz0 = T_in[ind_start+p-3] - T_ghost;
	func[ind_start+p-2] =  A[ind_start+p-2] * T_in[ind_start+p-3] + B[ind_start+p-2] * T_in[ind_start+p-2] + C[ind_start+p-2] * T_ghost
						+ D[ind_start+p-2] * Tz0 * Tz0;


}


__global__ void ADIZ1(double* Tsub, double* Tsub_int1, double* res_temp, int n, int m, int p, double mu_x, double mu_y, double Temp_amb, double Bi_new, double* muzA, double* muzB
, double* muzC, double* Tsub_ikj, double* Tsub_jki, double* Az_ijk_stored, double* Ay_jki_stored, double D_term, double E_term, model_parameters<double> m_paras)
{
	//== This routine gets replicated over every y & z direction.
	//===Note: z direction should only have a small number of points. 
	//===		Could use this to advantage.

	int idx = threadIdx.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	int ind_start, ind_c;

	__shared__ double s_veczx[adiz1_s_mem];
	__shared__ double s_matzx_sub[adiz1_s_mem];
	__shared__ double s_matzx_main[adiz1_s_mem];
	__shared__ double s_matzx_sup[adiz1_s_mem];

    if (idy < m && idz < (p-1))
	{
		while(idz < (p-1)){
			while(idy < m){
				// ind_start = idy*n + idz*n*(p-1);
				ind_start = 0;

				//== Loop through all components of shared memory.
				//---Note: # of threads is smaller than adiz1_s_mem, so we need to increment through.
				while(idx < n){
					//== Compute components of matrix/vector in shared memory
					get_mat_adizx(s_matzx_sub, s_matzx_main, s_matzx_sup, mu_x, ind_start, n, idx, m_paras.D_SW, m_paras.D_SE);
					get_vec_adizx(Tsub, res_temp, s_veczx, ind_start, idx, idy, idz, n, m, p, Temp_amb, Bi_new, mu_x, mu_y, muzA, muzB, muzC, Tsub_ikj, Tsub_jki, Az_ijk_stored, Ay_jki_stored, D_term, E_term, m_paras);
					idx += blockDim.x;
				}

				__syncthreads();

				idx = threadIdx.x;
				//== Make the last thread solve the linear system
				if(idx==(0)){
					tri_solve(s_matzx_sub, s_matzx_main, s_matzx_sup, s_veczx, n, ind_start, ind_start);

				}

				__syncthreads();

				//== Now each thread needs to transfer shared memory to global memory
				ind_c = idx+(idz)*n+idy*n*(p-1);
				while(idx < n){
					Tsub_int1[ind_c] = s_veczx[ind_start+idx];
					idx += blockDim.x;
					ind_c += blockDim.x;				
				}
				__syncthreads();

				// Move on to new indices
				idx = threadIdx.x;
				idy += blockDim.y * gridDim.y;
			}
		// Reset idx
			idy = threadIdx.y + blockIdx.y * blockDim.y;
			idz += blockDim.z * gridDim.z;
		}

	}
}

__global__ void ADIZ2(double* Tsub_int1, double* Tsub_int2, double* Tsub, double* res_temp, int n, int m, int p, double mu_y, double* Ay_jki_stored, model_parameters<double> m_paras)
{
	//== This routine gets replicated over every y & z direction.
	//===Note: z direction should only have a small number of points. 
	//===		Could use this to advantage.
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	int ind_start, ind_c;

	__shared__ double s_veczy[adiz2_s_mem];
	__shared__ double s_matzy_sub[adiz2_s_mem];
	__shared__ double s_matzy_main[adiz2_s_mem];
	__shared__ double s_matzy_sup[adiz2_s_mem];


    if (idx < n && idz < (p-1))
	{
		while(idz < p-1){
			while(idx < n){

				ind_start = 0;
				
				//== Form all matrices (LHS) and vectors (RHS)
				while(idy < m){
					get_mat_adizx(s_matzy_sub, s_matzy_main, s_matzy_sup, mu_y, ind_start, m, idy, m_paras.D_SS, m_paras.D_SN);
					get_vec_adizy(Tsub_int1, Tsub, res_temp, s_veczy, ind_start, idx, idy, idz, n, m, p, mu_y, Ay_jki_stored, m_paras);
					idy += blockDim.y;
				}

				__syncthreads();

				//== Reset index idy since it was just incremented above
				idy = threadIdx.y;

				//== Solve every N*(p-1) linear system of size MxM simultaneously
				if(idy==(0)){
					tri_solve(s_matzy_sub, s_matzy_main, s_matzy_sup, s_veczy, m, ind_start, ind_start);
				}

				__syncthreads();

				while(idy < m){

					ind_c = idy+(idz)*m+idx*m*(p-1);

					//== Transfer shared memory to global memory for intermediate Step T*
					// Tsub_int2[ind_c+1] = s_veczy[ind_start+idy];
					Tsub_int2[ind_c] = s_veczy[ind_start+idy];
					idy += blockDim.y;
				}

				__syncthreads();

				// Move on to new indices
				idy = threadIdx.y;
				idx += blockDim.x * gridDim.x;
			}
			// Reset idx
			idx = threadIdx.x + blockIdx.x * blockDim.x;
			idz += blockDim.z * gridDim.z;
		}

	}
}

__global__ void ADIZ3(double* Tsub_int2, double* Tsub, double* res_temp, int n, int m, int p, double* muzA, double* muzB, double* muzC, double Temp_amb, double Bi_new, double* Az_kij_stored
, double D_term, double E_term)
{
	//== This routine gets replicated over every y & z direction.
	//===Note: z direction should only have a small number of points. 
	//===		Could use this to advantage.
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int ind_start, ind_c;

	__shared__ double s_veczz[adiz3_s_mem];
	__shared__ double s_matzz_sub[adiz3_s_mem];
	__shared__ double s_matzz_main[adiz3_s_mem];
	__shared__ double s_matzz_sup[adiz3_s_mem];

    if (idx < n && idy < m)
    {
		while(idy < m){
			while(idx < n){

                ind_start = (p-1) * (threadIdx.x + (tPBx) * threadIdx.y);
				get_mat_adizz(s_matzz_sub, s_matzz_main, s_matzz_sup, p, ind_start, muzA, muzB, muzC, Bi_new, D_term);
				get_vec_adizz(Tsub_int2, Tsub, res_temp, s_veczz, ind_start, idx, idy, n, m, p, Temp_amb, Bi_new, muzA, muzB, muzC, Az_kij_stored, E_term);
				tri_solve(s_matzz_sub, s_matzz_main, s_matzz_sup, s_veczz, p-1, ind_start, ind_start);

				for (int k=0; k<(p-1); k++)
				{
					ind_c = k+idx*p+idy*n*p;
					//== Transfer shared memory to global memory for final step T^{n+1}
					Tsub[ind_c+1] = s_veczz[ind_start+k];
				}


					// Move on to new indices
					idx += blockDim.x * gridDim.x;
				}

				// Reset idx
				idx = threadIdx.x + blockIdx.x * blockDim.x;
				idy += blockDim.y * gridDim.y;
			}
	}
}

__device__ void get_mat_adizx(double* mat_sub, double* mat_main, double* mat_sup, double mu_in, int ind_start, int n_end, int i, double D_SW, double D_SE)
{

	//== Each thread should handle one case; thread id inputted as i
	if(i==0){
		mat_sub[ind_start+0] = 0;
		//== Robin BC at x=0
		mat_main[ind_start+0] = 1.0+2.0*mu_in-mu_in*D_SW;								//Note no flux boundary condition at x=0; T_{-1,j}=T_{0,j}
		mat_sup[ind_start+0] = -mu_in;
	} else if(i==(n_end-1)){
		mat_sub[ind_start + n_end-1] = -mu_in;
		//== Robin BC at x=L
		mat_main[ind_start + n_end-1] = 1.0+2.0*mu_in-mu_in*D_SE;						//Note no flux boundary condition at x=L, T_{n+1,j} = T_{n,j}
		mat_sup[ind_start + n_end-1] = 0;
	} else{
		mat_sub[ind_start+i] = -mu_in;
		mat_main[ind_start+i] = 1+2*mu_in;
		mat_sup[ind_start+i] = -mu_in;
	}

}

__device__ void get_vec_adizx(double* Tsub, double* res_temp, double* vecz, int ind_start, int i, int j, int k, int n, int m, int p, double Temp_amb, double Bi_new, double mu_x
, double mu_y, double* muzA, double* muzB, double* muzC, double* Tsub_ikj, double* Tsub_jki, double* Az_ijk_stored, double* Ay_jki_stored, double D_term, double E_term,  model_parameters<double> m_paras)
{
	int ind_c;
	int ind_jki = j+k*m+i*m*(p-1);
	int ind_ijk = i+j*n+k*n*m;

	int ind_ikj = i+(k)*n+j*n*(p-1);
	double dyy, dzz;

	ind_c = k+i*(p-1)+j*n*(p-1);
	dyy = DYY(Tsub_jki,i,j,k,n,m,p,mu_y, m_paras);

	Ay_jki_stored[ind_jki] = dyy;

	dzz = DZZ(Tsub,res_temp,i,j,k,n,p,Temp_amb,Bi_new,muzA,muzB,muzC,D_term,E_term);

	Az_ijk_stored[ind_ijk] = dzz;
	
	vecz[ind_start+i]= Tsub_ikj[ind_ikj] + DXX(Tsub_ikj,i,j,k,n,p,mu_x,m_paras) + 2.0*(dyy + dzz);

	if(i==0){
		vecz[ind_start+0] += mu_x*m_paras.E_SW;
	} else if(i==(n-1)){
		vecz[ind_start+n-1] += mu_x*m_paras.E_SE;
	}
}

__device__ void get_vec_adizy(double* Tsub_int1, double* Tsub, double* res_temp, double* vecz, int ind_start, int i, int j, int k, int n, int m, int p, double mu_y, double* Ay_jki_stored, model_parameters<double> m_paras)
{
	int ind_c;
	int ind_jki = j+k*m+i*m*(p-1);
	int ind_ikj = i+k*n+j*n*(p-1);
		
	//==Original
	ind_c = ind_ikj;

	vecz[ind_start+j]= Tsub_int1[ind_c] - Ay_jki_stored[ind_jki];

	//== Additions from Robin Type BCs in matrix portions
	if(j==0){
		vecz[ind_start+0] += mu_y * m_paras.E_SS;
	} else if(j==(m-1)){
		vecz[ind_start+m-1] += mu_y * m_paras.E_SN;
	}

}

__device__ void get_mat_adizz(double* mat_sub, double* mat_main, double* mat_sup, int p, int ind_start, double* muzA, double* muzB, double* muzC, double Bi_new, double D_term)
{
	mat_sub[ind_start+0] = 0;
	mat_main[ind_start+0] = 1 - muzB[0];
	mat_sup[ind_start+0] = -muzC[0];

    for(int k=1; k <= (p-3); k++){						
        mat_sub[ind_start+k] = -muzA[k];
        mat_main[ind_start+k] = 1 - muzB[k];
        mat_sup[ind_start+k] = -muzC[k];
	}

	mat_sub[ind_start+p - 2] = -muzA[p-2];
	mat_main[ind_start+p - 2] = 1 - muzB[p-2] - muzC[p-2] * D_term;//(1.0 - Bi_new) / (1.0 + Bi_new);
    mat_sup[ind_start+p - 2] = 0;
}

__device__ void get_vec_adizz(double* Tsub_int2, double* Tsub, double* res_temp, double* vecz, int ind_start, int i, int j, int n, int m, int p, double Temp_amb, double Bi_new 
, double* muzA, double* muzB, double* muzC, double* Az_kij_stored, double E_term)
{
	int ind_c;
	int ind_jki;
	for(int k=0;k<(p-1);k++){
		ind_c = 1+k+i*(p-1)+j*n*(p-1);
		ind_jki = j+(k)*m+i*m*(p-1); 

		vecz[ind_start+k]= Tsub_int2[ind_jki] - Az_kij_stored[ind_c-1];
	}

    vecz[ind_start + p-2] += muzC[p-2]*E_term;
    vecz[ind_start + 0] += muzA[0]*res_temp[i+j*n];
}

__device__ double DXX(double* Tsub, int i, int j, int k, int n, int p, double mu_x, model_parameters<double> m_paras)
{
    //== This function defines 1/2 K_2 T_{xx} = A_x/2
	double AX;
	int lm, l0, lp;
	double Ts_m, Ts_p;

	lm = (i-1)+k*n+j*n*(p-1);
	l0 = i+k*n+j*n*(p-1);
	lp = (i+1)+k*n+j*n*(p-1);

	Ts_m = Tsub[lm];
	Ts_p = Tsub[lp];

	if(i==0){

		//== Neumann BC
		lm = l0;

		//== Robin BC at x=0 (southwest)
		Ts_m = m_paras.D_SE * Tsub[l0] + m_paras.E_SE;

	} else if(i==(n-1)){

		//== Neumann BC
		lp = l0;

		//== Robin BC at x=L (southeast)
		Ts_p = m_paras.D_SW * Tsub[l0] + m_paras.E_SW;
	}

	// AX = mu_x * ( Tsub[lm] - 2.0 * Tsub[l0] + Tsub[lp] );
	AX = mu_x * ( Ts_m - 2.0 * Tsub[l0] + Ts_p );
	return AX;
}

__device__ double DYY(double* Tsub, int i, int j, int k, int n, int m, int p, double mu_y, model_parameters<double> m_paras)
{
    //== This function defines 1/2 K_2 T_{yy} = A_y/2
	double AY;
	int lm, l0, lp;

	double Ts_m, Ts_p;

	l0 = j+k*m+i*m*(p-1);
	lm = (j-1)+k*m+i*m*(p-1);
	lp = (j+1)+k*m+i*m*(p-1);

	Ts_m = Tsub[lm];
	Ts_p = Tsub[lp];
	if(j==0){
		//== Neumann BC
		lm = l0;

		//== Robin BC at y=0 (SouthSouth)
		Ts_m = m_paras.D_SS * Tsub[l0] + m_paras.E_SS;

	} else if(j==(m-1)){
		//== Neumann BC
		lp = l0;

		//== Robin BC at y=L (SouthNorth)
		Ts_p = m_paras.D_SN * Tsub[l0] + m_paras.E_SN;
	}

	AY = mu_y * ( Ts_m - 2.0 * Tsub[l0] + Ts_p );
	return AY;
}

__device__ double DZZ(double* Tsub, double* res_temp, int i, int j, int k, int n, int p, double Temp_amb, double Bi_new, double* muzA, double* muzB, double* muzC, double D_term, double E_term)
{
    //== This function defines 1/2 K_2 T_{zz} approximation
	double T_ghost;
	int ind_c, ind_p, ind_pp;
	double AZ;

	ind_c = k+j*n*p + i*p;
	ind_p = ind_c + 1;
	ind_pp = ind_p + 1;

	if(k==0){
		//== Starting at k=1 index since liquid-solid interface is set
		//== May need to edit this term since we don't get two portions
		AZ = muzA[0]*( Tsub[ind_c] ) + muzB[0] * Tsub[ind_p] + muzC[0] * Tsub[ind_pp];
	} else if(k==(p-2)){
		T_ghost = D_term*Tsub[ind_p] + E_term;
		AZ = muzA[p-2] * Tsub[ind_c] + muzB[p-2]*Tsub[ind_p] + 1.0*muzC[p-2]*T_ghost;
	} else{
		AZ = muzA[k]*Tsub[ind_c]+ muzB[k]*Tsub[ind_p]+muzC[k]*Tsub[ind_pp];
	}

	return AZ;
}

__global__ void transpose_sub_1(double* Tsub, double* Tsub_ikj, double* Tsub_jki, int n, int m, int p)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	int ind_kij, ind_ikj, ind_jki;
	double Tsub0;

	//== Naive, simple transpose
	if(idx < n && idy < m && idz < (p-1))
	{
		ind_jki = idy + idz*m + idx*m*(p-1);

		ind_kij = idz + idx*p + idy*n*p;
		ind_ikj = idx + idz*n + idy*n*(p-1);
		Tsub0 = Tsub[ind_kij+1];
		Tsub_ikj[ind_ikj] = Tsub0;
		Tsub_jki[ind_jki] = Tsub0;
	}
}

__global__ void transpose_A_mats(double* Az_ijk, double* Az_kij, double* Tsub_int1_ijk, double* Tsub_int1_jki, int n, int m, int p)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	int ind_kij, ind_ikj, ind_ijk, ind_jki, ind_jik;
	double reg_var, reg_var_y;

	//== Naive, simple transpose
	if(idx < n && idy < m && idz < (p-1))
	{
		ind_ijk = idx + idy*n + idz*n*m;
		ind_jki = idy + idz*m + idx*m*(p-1);

		ind_kij = idz + idx*(p-1) + idy*n*(p-1);
		reg_var = Az_ijk[ind_ijk];
		Az_kij[ind_kij] = reg_var;

	}
}

__device__ void sub_newton(double &x_in, double Tpm1, double Bi_new, double Temp_amb, double Temp_scl, int TC_model)
{
	//== Description: Use Newton's method to solve nonlinear BC at z=-H_s: k dT/dz = Bi (T-T_a)
	int max_newt_iter;
	double G, Gprime, x_guess, newt_tol;
	bool newt_completed;

	//== Set the tolerance
	newt_tol = 1e-10;
	max_newt_iter = 100;
	newt_completed=false;

	x_guess = Tpm1;
	//== Initialize x_in
	newt_G(x_guess, Tpm1, G, Gprime, Bi_new, Temp_amb, Temp_scl, TC_model);

	for(int k=1;k<=max_newt_iter;k++){
		x_guess = x_guess - G / Gprime;
		newt_G(x_guess, Tpm1, G, Gprime, Bi_new, Temp_amb, Temp_scl, TC_model);
		if(abs(G)<newt_tol)
		{
			newt_completed=true;
			break;
		}
	}

	//== Set exact solution
	if(newt_completed){
		x_in =  x_guess;
	} else{
		printf("Newton solve for ghost point Tp not solved in iter max");
	}

}

__device__ void newt_G(double Tp, double Tpm1, double &G, double &Gp, double Bi_new, double Temp_amb, double Temp_scl, int TC_model)
{
	double Tp_avg, Tpz, TC, dTC, ddTC;

	Tp_avg = 0.5 * (Tp + Tpm1);
	Tpz = (Tpm1 - Tp);

	//== Get k(T) and k'(T) using T=T_avg at bottom boundary
	get_TC_funcs(Tp_avg, &TC, &dTC, &ddTC, Temp_scl, TC_model);

	//== Give function G for G(Tp, T_{p-1}) = 0
	//G = TC * (Tpm1 - Tp) * inv_dzs_var - Bi * (Tp_avg)
	G = TC * Tpz - Bi_new * (Tp + Tpm1) + 2.0*Bi_new*Temp_amb;
	
	//== Give derivative of G w.r.t. T_{p-1}
	// Gp = TC * (-inv_dzs_var) + 0.5d0 * Tpz * dTC - Bi/2.0d0
	Gp = -TC + 0.5 * Tpz * dTC - Bi_new;

	//== Now we have g(x) and g'(x); next update x

}

#endif
