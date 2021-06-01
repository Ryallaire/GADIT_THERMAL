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

#ifndef MODEL_THERMAL1
#define MODEL_THERMAL1

// Imports base model_parameters template.
#include "solver_template.h"
#include <math.h>

namespace model_thermal1
{
	// If copying to make new model, change to new model id.
	const model::id ID = model::THERMAL1;

	// Definition of nonlinear functions for this model ID.
	template <typename DATATYPE, model::id MODEL_ID = ID> __device__ __forceinline__
		void nonlinear_functions(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h, model_parameters<DATATYPE> modelParas, int h_index)
	{

		DATATYPE cC = modelParas.cC;
		DATATYPE cN = modelParas.cN;
		DATATYPE cK = modelParas.cK;

		DATATYPE cD = modelParas.cD;
		DATATYPE beta = modelParas.beta;
		DATATYPE b = modelParas.b;
		DATATYPE w = modelParas.w;

		DATATYPE invW = modelParas.inv_w;
		DATATYPE beta2 = modelParas.beta2;
		DATATYPE w2 = modelParas.w2;
		DATATYPE two_w = modelParas.two_w;
		DATATYPE three_b = modelParas.three_b;
		DATATYPE scaled_cN = modelParas.scaled_cN;
		//DATATYPE cK_b2 = modelParas.cK_b2;
		//DATATYPE scaled_cK_b3 = modelParas.scaled_cK_b3;
		DATATYPE cK_b2 = cK*b*b;
		DATATYPE scaled_cK_b3 = 3.0*cK*b*b*b;

		//Redefine gamma terms
		DATATYPE gamma_0 = modelParas.gamma_0;
		DATATYPE gamma_T = modelParas.gamma_T;
		DATATYPE T_0 = modelParas.Temp_melt;		// Set reference temperature to melting temperature


		DATATYPE h = d_h[h_index];
		DATATYPE h2 = h*h;
		DATATYPE h3 = h2*h;

		d_ws.f1[h_index] = cC*h3;				// Capillary Term
		d_ws.df1[h_index] = 3.0*cC*h2;			// Derivative of capillary term

		DATATYPE tau = tanh((h - 2.0 * b)*invW);
		DATATYPE kappa = tau - 1.0;;

		DATATYPE eta = h2 + beta2;
		DATATYPE eta2 = eta*eta;
		DATATYPE invEta = 1.0 / eta;

		DATATYPE shareNematic = scaled_cN*(tau + 1.0)*(tau + 1.0)*invEta*invEta*invEta*h3;
		DATATYPE invH = 1.0 / h;

		d_ws.f2[h_index] = -h*shareNematic*(h2*two_w - w*eta + h*eta*kappa);			// Nematic Term + Disjoining Pressure
		d_ws.f2[h_index] += cK_b2*(2.0 - three_b*invH);

		d_ws.df2[h_index] = invW*invEta*shareNematic*(4.0 * w2*(3.0 * h2*h2 - 4.0 * h2*eta + eta2) + h*eta*(8.0 * h2*w + 2.0 * h*eta - 7.0 * w*eta)*kappa + 3.0 * h2*eta2*kappa*kappa);
		d_ws.df2[h_index] += scaled_cK_b3*invH*invH;	// Derivative of NT + DP

		// Marangoni term.
		d_ws.f3[h_index] = cD*h2;
		d_ws.df3[h_index] = 2.0*cD*h;
	}


}

#endif
