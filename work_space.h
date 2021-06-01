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

#ifndef WORK_SPACE
#define WORK_SPACE

#include "cuda_runtime.h"
#include "memory_manager.h"
#include "cuda_parameters.h"
#include "dimensions.h"

template <typename DATATYPE>
struct __align__(16) PentaLMatrix
{
	DATATYPE _a;
	DATATYPE _b;
};

template <typename DATATYPE>
struct __align__(16) PentaUMatrix
{
	DATATYPE _c;
	DATATYPE _d;
	DATATYPE _e;
};


template <typename DATATYPE>
struct PentaLUMatrix
{
	PentaLMatrix<DATATYPE> *_L;
	PentaUMatrix<DATATYPE> *_U;
	DATATYPE *_f;
};




template <typename DATATYPE>struct penta_diag_row
{

	DATATYPE a;
	DATATYPE b;
	DATATYPE c;
	DATATYPE d;
	DATATYPE e;
	DATATYPE f;

};

//////////////////////////////////////////////////////////////////////////////////////////////////////
//																									//
// Description: 
//		Structure to hold global memory device pointers, simplifying function calls.
//
//		Works in conjunction 'memory_manager.h', which facilitate memory allocation and
//		data transfer between host (CPU) and device (GPU). 
//
//		A smaller version of "unified_work_space" (contained here) which,
//			1) reduces the size of the object passed to kernel;
//			2) reduces the degree of pointer separation between passed object
//			   and reference location created by 'memory_manager.h.
// WARNING: 
//		Must be passed by values. Passing by pointer will result in invalid pointers.
//																									//
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename DATATYPE> struct reduced_device_workspace
{

	// Variables used to compute the linear system to solve at each newton iteration;

	DATATYPE *h;
	DATATYPE *h_guess;

	DATATYPE *f1;
	DATATYPE *df1;

	DATATYPE *f2;
	DATATYPE *df2;
	
	DATATYPE *f3;
	DATATYPE *df3;

	DATATYPE *F_fixed;
	DATATYPE *F;
	DATATYPE *F_err;
	DATATYPE *F_fixed_stored;
	DATATYPE *F_guess_stored;

	DATATYPE *v;
	DATATYPE *w_transpose;

	DATATYPE *Jx_a;
	DATATYPE *Jx_b;
	DATATYPE *Jx_c;
	DATATYPE *Jx_d;
	DATATYPE *Jx_e;

	DATATYPE *Jy_a;
	DATATYPE *Jy_b;
	DATATYPE *Jy_c;
	DATATYPE *Jy_d;
	DATATYPE *Jy_e;

	// DATATYPE *Jy_a_stored;
	// DATATYPE *Jy_b_stored;
	// DATATYPE *Jy_c_stored;
	// DATATYPE *Jy_d_stored;
	// DATATYPE *Jy_e_stored;

	// Variables used solve penta diagonal matrix
	DATATYPE *LU_a;
	DATATYPE *LU_b;
	DATATYPE *LU_c;
	DATATYPE *LU_d;
	DATATYPE *LU_e;

	char *solution_flags;
	
	// Variables used for Thermal model
	DATATYPE *gamma;
	DATATYPE *visc_rat;

};



//////////////////////////////////////////////////////////////////////////////////////////////////////
//																									//
// Description: 
//		Class that handles 
//			1) total memory used on host (CPU),
//			2) and total global memory on device (GPU).
//
//		Serves three (3) purposes:
//			1) Declare variables and storage location(s), handled by 'memory_unit';
//			2) Allocate and deallocated memory, simplified using 'memory_manager.h';
//			3) With 'reduced_device_workspace', pointer reference value may be copied 
//			   (from 'memory_unit') to:
//					a) reduce total memory usage,
//					b) remove reference value passing at each newton iteration and time step,
//					c) and provide one object to pass, 'reduced_device_workspace' while 
//					   maintaining a robust set of parameters.										//
//																									//
// WARNING:																							//
//		Care must be taken when linking 'memory_manager.h' and 'reduced_device_workspace'			//
//																									//
//////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace memory_manager;

template<typename DATATYPE> class unified_work_space
{
//
public:
	reduced_device_workspace<DATATYPE> reduced_dev_ws;


	memory_unit<DATATYPE> *h;
	memory_unit<DATATYPE> *x;
	memory_unit<DATATYPE> *y;

	memory_unit<DATATYPE>  *h_guess;

	memory_unit<DATATYPE>  *f1;
	memory_unit<DATATYPE>  *df1;

	memory_unit<DATATYPE>  *f2;
	memory_unit<DATATYPE>  *df2;
	
	memory_unit<DATATYPE>  *f3;
	memory_unit<DATATYPE>  *df3;

	memory_unit<DATATYPE>  *F_fixed;
	memory_unit<DATATYPE>  *F;
	memory_unit<DATATYPE>  *F_err;
	memory_unit<DATATYPE>  *F_fixed_stored;
	memory_unit<DATATYPE>  *F_guess_stored;

	memory_unit<DATATYPE>  *J_a;
	memory_unit<DATATYPE>  *J_b;
	memory_unit<DATATYPE>  *J_c;
	memory_unit<DATATYPE>  *J_d;
	memory_unit<DATATYPE>  *J_e;

	// memory_unit<DATATYPE>  *J_a_stored;
	// memory_unit<DATATYPE>  *J_b_stored;
	// memory_unit<DATATYPE>  *J_c_stored;
	// memory_unit<DATATYPE>  *J_d_stored;
	// memory_unit<DATATYPE>  *J_e_stored;


	memory_unit<DATATYPE>  *w_transpose;
	memory_unit<char> *solution_flags;
	
	memory_unit<DATATYPE>  *gamma;			//Surface tension for Thermal Model
	memory_unit<DATATYPE>  *visc_rat;		//Dimensionless viscosity

	//== Add memory units for heat code
	//== Create pointers to device memory units
	/*memory_unit<DATATYPE>  *res_temp;
	memory_unit<DATATYPE>  *res_temp_trans;
	memory_unit<DATATYPE>  *res_temp_prev;
	memory_unit<DATATYPE>  *res_temp_trans_prev;
	memory_unit<DATATYPE>  *Tsub;
	memory_unit<DATATYPE>  *T_int;
	memory_unit<DATATYPE>  *T_int_prev;
	memory_unit<DATATYPE>  *courant;
	memory_unit<DATATYPE>  *res_trans;
	memory_unit<DATATYPE>  *res_nxt_trans;
	memory_unit<DATATYPE>  *err_local;
	memory_unit<DATATYPE>  *res_prev; */


	unified_work_space(){};
	
	void initalize_memory(dimensions dims)
	{ 

		//Initializing memory on host & device with memory_unit 
		h = new memory_unit<DATATYPE>(memory_scope::PINNED, dims.n_pad, dims.m_pad);

		x = new memory_unit<DATATYPE>(memory_scope::HOST_ONLY, dims.n);
		y = new memory_unit<DATATYPE>(memory_scope::HOST_ONLY, dims.m);

		h_guess = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n_pad, dims.m_pad);


		f1 = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n_pad, dims.m_pad);
		df1 = new memory_unit<DATATYPE>(f1);
		f2 = new memory_unit<DATATYPE>(f1);
		df2 = new memory_unit<DATATYPE>(f1);
		f3 = new memory_unit<DATATYPE>(f1);
		df3 = new memory_unit<DATATYPE>(f1);
		

		F = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		F_fixed = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		F_err = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		F_fixed_stored = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		F_guess_stored = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);

		J_a = new memory_unit<DATATYPE>(F);
		J_b = new memory_unit<DATATYPE>(F);
		J_d = new memory_unit<DATATYPE>(F);
		J_c = new memory_unit<DATATYPE>(F);
		J_e = new memory_unit<DATATYPE>(F);

		// J_a_stored = new memory_unit<DATATYPE>(F);
		// J_b_stored = new memory_unit<DATATYPE>(F);
		// J_d_stored = new memory_unit<DATATYPE>(F);
		// J_c_stored = new memory_unit<DATATYPE>(F);
		// J_e_stored = new memory_unit<DATATYPE>(F);

		w_transpose = new memory_unit<DATATYPE>(F);

		solution_flags = new memory_unit<char>(memory_scope::PINNED, dims.n_reduced , dims.m_reduced);

		gamma = new memory_unit<DATATYPE>(h);		//Michael Suggestion 1
		visc_rat = new memory_unit<DATATYPE>(h);

		//== Allocate Memory for Heat Variables on Device
		
		/*res_temp = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		res_temp_trans = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		res_temp_prev = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		res_temp_trans_prev = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		Tsub = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m, dims.p);
		T_int = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		T_int_prev = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		courant = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, 5);								//== Only 5 courant numbers; need on device
		res_trans = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n_pad, dims.m_pad);
		res_nxt_trans = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n_pad, dims.m_pad);
		err_local = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n, dims.m);
		res_prev = new memory_unit<DATATYPE>(memory_scope::DEVICE_ONLY, dims.n_pad, dims.m_pad); */

		initiateVaribles(h, h_guess);
		initiateVaribles(x, y);
		initiateVaribles(f1, df1, f2, df2, f3, df3);
		initiateVaribles(J_a, J_b, J_c, J_d, J_e, F, F_fixed, F_err);
		initiateVaribles(w_transpose);
		initiateVaribles(solution_flags);
		initiateVaribles(gamma, visc_rat);			// Initializes memory insize gpu/cpu for gamma
		initiateVaribles(F_fixed_stored, F_guess_stored);
		// initiateVaribles(J_a_stored, J_b_stored, J_c_stored, J_d_stored, J_e_stored);
		/*
		initiateVaribles(res_temp, res_temp,trans, res_temp_prev, res_temp_trans_prev);
		initiateVaribles(Tsub, T_int, T_int_prev);
		initiateVaribles(courant);
		initiateVaribles(res_trans, res_nxt_trans);
		initiateVaribles(err_local, res_prev); */

		// unique allocations
		reduced_dev_ws.h = h->data_device;
		reduced_dev_ws.h_guess = h_guess->data_device;

		reduced_dev_ws.f1 = f1->data_device;
		reduced_dev_ws.df1 = df1->data_device;

		reduced_dev_ws.f2 = f2->data_device;
		reduced_dev_ws.df2 = df2->data_device;
		
		reduced_dev_ws.f3 = f3->data_device;
		reduced_dev_ws.df3 = df3->data_device;

		reduced_dev_ws.F_fixed = F_fixed->data_device;
		reduced_dev_ws.F_err = F_err->data_device;

		reduced_dev_ws.F_fixed_stored = F_fixed_stored->data_device;
		reduced_dev_ws.F_guess_stored = F_guess_stored->data_device;

		reduced_dev_ws.w_transpose = w_transpose->data_device;

		reduced_dev_ws.F = F->data_device;
		// save memory since new F requires v be added to h_guess
		reduced_dev_ws.v = F->data_device;


		// matrices share same memory space
		reduced_dev_ws.Jx_a = J_a->data_device;
		reduced_dev_ws.Jx_b = J_b->data_device;
		reduced_dev_ws.Jx_c = J_c->data_device;
		reduced_dev_ws.Jx_d = J_d->data_device;
		reduced_dev_ws.Jx_e = J_e->data_device;

		reduced_dev_ws.Jy_a = J_a->data_device;
		reduced_dev_ws.Jy_b = J_b->data_device;
		reduced_dev_ws.Jy_c = J_c->data_device;
		reduced_dev_ws.Jy_d = J_d->data_device;
		reduced_dev_ws.Jy_e = J_e->data_device;

		reduced_dev_ws.LU_a = J_a->data_device;
		reduced_dev_ws.LU_b = J_b->data_device;
		reduced_dev_ws.LU_c = J_c->data_device;
		reduced_dev_ws.LU_d = J_d->data_device;
		reduced_dev_ws.LU_e = J_e->data_device;


		reduced_dev_ws.solution_flags = solution_flags->data_device;
		
		// Add in surface tension to reduced device workspace for access in d_ws
		reduced_dev_ws.gamma = gamma->data_device;
		reduced_dev_ws.visc_rat = visc_rat->data_device;

	}

	void clean_workspace()
	{
		freeAll(h, h_guess);
		freeAll(x, y);
		freeAll(f1, df1, f2, df2, f3, df3);
		freeAll(J_a, J_b, J_c, J_d, J_e, F, F_fixed, F_err);
		freeAll(w_transpose);
		freeAll(solution_flags);
		freeAll(gamma, visc_rat);
		freeAll(F_fixed_stored, F_guess_stored);
	}
};

#endif