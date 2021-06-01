// Attempting to Solve Heat Equation
//
//
//
//
//
// ----------------------------------------------------------------


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
// Name:		change_mat_par.h
// Version: 	1.0
// Purpose:		Changes the material parameters (surface tension and viscosity),
//				based on spatial average of temperature bar{T(x,y,z,t)}.
// CUDA Info:	Program, for now, uses CPU to solve the temperature. Information must
//				first be transferred, for film thickness, h, from GPU to CPU, and inputted
//				as variable, res. Once heat solution is complete, must update surface
//				tension, gamma(T)=gamma_0 +gamma_T(T-T_0). This term is then sent
//				to the gpu for solution of thin film equation.
// ----------------------------------------------------------------------------------
#ifndef CHANGE_MAT_PAR
#define CHANGE_MAT_PAR

#include "work_space.h"
#include "parameters.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include "solver_template.h"
#include <math.h>

void change_mat_par(double temp_avg, model_parameters<double> &m_paras)
{
	double visc_ratio;
	double surf_tens_in;
	double visc_in;

	
	if(m_paras.time_varying_surf_tens)
	{
		surf_tens_in = m_paras.gamma_0 + m_paras.gamma_T*m_paras.T_scl*(temp_avg - m_paras.Temp_melt);
	} 
	else 
	{
		surf_tens_in = m_paras.gamma_0;
	}

	if(m_paras.time_varying_visc)
	{
		visc_in = m_paras.visc_0*exp(m_paras.E_act / (m_paras.R*m_paras.T_scl)*(1.0 / temp_avg-1.0/(m_paras.Temp_melt)));
	} 
	else 
	{
		visc_in = m_paras.visc_0;
	}

	if(!m_paras.constant_parameters)
	{
		visc_ratio = m_paras.visc_0 / visc_in;
		m_paras.cC = visc_ratio * surf_tens_in / m_paras.gamma_0;
		m_paras.cK = m_paras.kappa_dp*m_paras.l_scl / m_paras.gamma_0 * visc_ratio;
	}
	//surf_tens_in = m_paras.gamma_0; // + m_paras.gamma_T*m_paras.T_scl*(temp_avg - m_paras.Temp_melt);
	//visc_in = m_paras.visc_0*exp(m_paras.E_act / m_paras.R*(1.0 / (temp_avg*m_paras.T_scl)-1.0/(m_paras.Temp_melt*m_paras.T_scl)));

	//visc_ratio = m_paras.visc_0 / visc_in;
	//m_paras.cC = visc_ratio * surf_tens_in / m_paras.gamma_0;
	//m_paras.cK = m_paras.kappa_dp*m_paras.l_scl / m_paras.gamma_0 * visc_ratio;

}


#endif
