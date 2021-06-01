
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

#ifndef PARAMETERS
#define PARAMETERS


#include <chrono>
#include <string>
#include <math.h>
#include <cmath>

namespace newton_status{
	enum status
	{
		SUCCESS = 0,
		CONVERGENCE_FAILURE_SMALL = 1,
		CONVERGENCE_FAILURE_LARGE = 2,
		NEGATIVE_SOLUTION = 3,
		TRUNCATION_ERROR = 4,
		TEMP_CHANGE_DT = 5,
		INCREASE_DT = 255,
		FROZEN = 6,
	};
}

namespace newton_stage
{
	enum stage
	{
		INITIAL,
		PRELOOP,
		LOOP,
	};
}

enum bc_postition
{
	FIRST, SECOND, THIRD, FIRST_LAST, SECOND_LAST, THIRD_LAST, INTERIOR,
};


enum boundary_condtion_type{
	SYMMETRIC,
	CONSTANT_FLOW,
};

namespace format_parameter_output
{
	std::string make_title(std::string title)
	{

		std::string output;
		std::string border;

		size_t length = title.length();

		border = std::string(length + 4, '-');


		output = border + "\n";
		output += "- " + title + " -\n";
		output += border + "\n";

		return output;

	}

	std::string datatype(int val)
	{
		std::string output;
		char buff[100];
		sprintf(buff, "%d" , val);
		output = buff;
		return output;
	}

	std::string datatype(size_t val)
	{
		std::string output;
		char buff[100];
		sprintf(buff, "%llu", val);
		output = buff;
		return output;
	}
	// std::string  datatype(double val)
	template <typename DATATYPE> std::string datatype(DATATYPE val)
	//std::string  datatype(double val)
	{

		std::string output;
		char buff[100];
		if ( val == 0 )
			sprintf(buff, "%i", 0.0);
		else if (abs(val) < pow(10, -6))
			sprintf(buff, "%e", val);
		else if(abs(val) > pow(10, 6))
			sprintf(buff, "%e", val);
		else
			sprintf(buff, "%f", val);
		
		output = buff;
		return output;
	}

	/* std::string  datatype(float val)
	{

		std::string output;
		char buff[100];
		if (val == 0)
			sprintf(buff, "%i", 0.0);
		else if (abs(val) < pow(10, -6))
			sprintf(buff, "%e", val);
		else if (abs(val) > pow(10, 6))
			sprintf(buff, "%e", val);
		else
			sprintf(buff, "%f", val);

		output = buff;
		return output;
	} */

}

template <typename DATATYPE> struct spatial_parameters{

	DATATYPE ds;
	DATATYPE dx;
	DATATYPE dy;
	DATATYPE dx2;
	DATATYPE dy2;
	DATATYPE dzs;
	DATATYPE dzs2;
	DATATYPE dzs2t2;
	DATATYPE dzst2;
	DATATYPE dxt2;
	DATATYPE dyt2;
	DATATYPE inv_dx2;
	DATATYPE inv_dy2;
	DATATYPE inv_dxt2;
	DATATYPE inv_dyt2;
	DATATYPE inv_dzst2;
	DATATYPE inv_dx2t2;
	DATATYPE inv_dy2t2;
	DATATYPE inv_dzs2t2;
	
	DATATYPE inv_ds;
	DATATYPE inv_ds2;
	DATATYPE inv_ds4;


	DATATYPE scaled_inv_ds2;
	DATATYPE scaled_inv_ds4;
	
	size_t n;
	size_t m;
	size_t Nzs;
	size_t p;
	size_t padding;
	size_t n_pad;
	size_t m_pad;

	DATATYPE x0;
	DATATYPE xn;
	DATATYPE y0;
	DATATYPE ym;
	DATATYPE th_sio2;

	void compute_derived_parameters()
	{
		this->xn = ds*double(n);
		this->ym = ds*double(m);

		dx = ds;									
		dy = ds;
		dzs = th_sio2 / (Nzs - 0.5);				//== Note that (Nzs should be converted to double first)
		dzs2	=	dzs*dzs;						// dzs2=dzs^2 (substrate z spacing)
		dzs2t2	=	dzs2*2.0;
		dzst2	=	dzs*2.0;
		dyt2 = dy * 2.0;
		dxt2 = dx * 2.0;
		dx2 = dx * dx;
		dy2 = dy * dy;
		
		inv_ds = 1.0 / ds;
		inv_ds2 = inv_ds*inv_ds;
		inv_ds4 = inv_ds2*inv_ds2;

		inv_dx2 = 1.0 / dx2;
		inv_dy2 = 1.0 / dy2;
		inv_dxt2 = 0.5*inv_ds;
		inv_dyt2 = 0.5*inv_ds;
		inv_dzst2 = 1.0 / dzst2;
		inv_dx2t2 = 0.5*inv_dx2;
		inv_dy2t2 = 0.5*inv_dy2;
		inv_dzs2t2 = 1.0 / dzs2t2;

		n_pad = n + 2 * padding;
		m_pad = m + 2 * padding;

		// absorbing all constants into derivative factors

		// 1) 1/2 factor from evaluating functions at half points i.e h(x-dx/2) = ( h(x)+h(x-dx) )/2 + O(dx^2)
		// 2) another 1/2 factor from Crank-Nicholson i.e theta and 1-theta for theta=1/2
		DATATYPE const prefactor = 0.5*0.5;

		scaled_inv_ds2 = prefactor*inv_ds2;
		scaled_inv_ds4 = prefactor*inv_ds4;
	}

	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Spatial");

		output += "ds = " + format_parameter_output::datatype(this->ds) + "\n";
		output += "n  = " + format_parameter_output::datatype(this->n) + "\n";
		output += "x0 = " + format_parameter_output::datatype(this->x0) + "\n";
		output += "xn = " + format_parameter_output::datatype(this->xn) + "\n";
		output += "m  = " + format_parameter_output::datatype(this->m) + "\n";
		output += "y0 = " + format_parameter_output::datatype(this->y0) + "\n";
		output += "ym = " + format_parameter_output::datatype(this->ym) + "\n";
		output += "Nzs  = " + format_parameter_output::datatype(this->Nzs) + "\n";
		output += "\n";
		return output;
	}

};



template <typename DATATYPE> struct newton_parameters{
	DATATYPE error_tolerence;
	DATATYPE small_value;
	DATATYPE truncation_tolerance;
	
	int max_iterations;
	int min_iterations;	
	
	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Newton Iterations");

		output += "error_tolerence = " + format_parameter_output::datatype(this->error_tolerence) + "\n";
		output += "max_iterations  = " + format_parameter_output::datatype(this->max_iterations) + "\n";
		output += "min_iterations  = " + format_parameter_output::datatype(this->min_iterations) + "\n";
		output += "\n";

		return output;
	}

};

struct io_parameters
{
	std::string root_directory;
	bool is_full_text_output;
	bool is_console_output;
};

template <typename DATATYPE> struct temporal_parameters{
	
	DATATYPE t_start;
	DATATYPE t_end;

	DATATYPE dt_out;
	
	DATATYPE dt_min;
	DATATYPE dt_max;

	DATATYPE dt_init;
	
	DATATYPE dt_ratio_increase;
	DATATYPE dt_ratio_decrease;
	
	DATATYPE dt_thermal;			//Set Minimum Update Time for Heat Solve

	int min_stable_step;

	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Temporal");

		output += "t_start           = " + format_parameter_output::datatype(this->t_start) + "\n";
		output += "t_end             = " + format_parameter_output::datatype(this->t_end) + "\n";
		output += "dt_out            = " + format_parameter_output::datatype(this->dt_out) + "\n";
		output += "dt_min            = " + format_parameter_output::datatype(this->dt_min) + "\n";
		output += "dt_max            = " + format_parameter_output::datatype(this->dt_max) + "\n";
		output += "dt_init           = " + format_parameter_output::datatype(this->dt_init) + "\n";
		output += "min_stable_step   = " + format_parameter_output::datatype(this->min_stable_step) + "\n";
		output += "dt_ratio_increase = " + format_parameter_output::datatype(this->dt_ratio_increase) + "\n";
		output += "dt_ratio_decrease = " + format_parameter_output::datatype(this->dt_ratio_decrease) + "\n";
		output += "\n";

		return output;
	}

};

 struct backup_parameters{
	long long updateTime ;
	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Backup");

		output += "updateTime = " + format_parameter_output::datatype(this->updateTime) + "\n";
		output += "\n";

		return output;
	}
	
};

template <typename DATATYPE> struct initial_parameters{
			int nx;
			int ny;
			DATATYPE epy;
			DATATYPE epx;
			DATATYPE h0;

			std::string to_string()
			{

				std::string output;
				output = format_parameter_output::make_title("Linear Wave Initial Condition");

				output += "h0  = " + format_parameter_output::datatype(this->h0) + "\n";
				output += "nx  = " + format_parameter_output::datatype(this->nx) + "\n";
				output += "ny  = " + format_parameter_output::datatype(this->ny) + "\n";
				output += "epx = " + format_parameter_output::datatype(this->epx) + "\n";
				output += "epy = " + format_parameter_output::datatype(this->epy) + "\n";
				output += "\n";

				return output;
			}
};

	template <typename DATATYPE > struct model_parameters{
		// Main parameters
		DATATYPE cK;
		DATATYPE cC;
		DATATYPE cN;
		DATATYPE cD;					//Marangoni Term-- need to define!!
		DATATYPE beta;
		DATATYPE b;
		DATATYPE w;
		DATATYPE theta_c;
		DATATYPE kappa_dp;
		DATATYPE vdw_n;
		DATATYPE vdw_m;
		DATATYPE cap_M;

		// Scales
		DATATYPE l_scl;					// Lateral Length Scales						(m)
		DATATYPE l_scl2;
		DATATYPE h_scl;					// Transverse Length Scale
		DATATYPE t_scl;					// Time Scale									(s)
		DATATYPE T_scl;					// Temperature Scale							(K)

		// Material Parameters
		DATATYPE gamma_0;				// Surface Tension at reference temp			(J/m^2)
		DATATYPE gamma_T;				// dgamma/dT (constant for linear profile)		(J/m^2/K)
		DATATYPE gamma_ratio;			// Ratio of gamma_T to gamma_0					(1)
		DATATYPE rho_sio2;				// Density of Substrate Oxide Layer   			(kg/m^3)
		DATATYPE rho_m;					// Density of Metal								(kg/m^3)
		DATATYPE visc_0;				// Viscosity of Metal at melting temperature	(Pa s)
		DATATYPE Ceff_sio2;				// Effective Heat Capacity of Oxide Layer		(J/kg/K)
		DATATYPE Ceff_m;				// Effective Heat Capacity of Metal				(J/kg/K)
		DATATYPE k_sio2;				// Thermal Conductivity of Oxide Layer			(W/m/K)
		DATATYPE k_m;					// Thermal Conductivity of Metal				(W/m/K)
		DATATYPE k_ratio;
		DATATYPE Temp_room;				// Room Temperature								(K)
		DATATYPE Temp_melt;				// Melting Temperature of Metal					(K)
		DATATYPE Temp_amb;				// Scaled ambient temperature
		DATATYPE surf_tens;				// Variable surface tension
		DATATYPE visc;					// Variable Viscosity
		DATATYPE th_sio2;				// Dimensionless Substrate Thickness

		// Source Parameters
		DATATYPE energy_0;				// Laser Fluence (energy density)				(J/m^2)
		DATATYPE alpha_m_inv;			// Metal Absorption Length						(m)
		DATATYPE alpha_r_inv;			// Metal Reflective Length						(m)
		DATATYPE reflect_coef;			// Reflective coefficient						(Dimensionless)
		DATATYPE zeta;					// Renormalization Factor						(Dimensionless)
		DATATYPE tp;					// Peak Time for Gaussian Pulse					(s)
		DATATYPE sigma;					// Pulse Standard Deviation						(s)
		DATATYPE E_act;					// Activation Energy for the film
		DATATYPE R;						// Universal Gas Constant
		DATATYPE Q_coef;				// Coefficient in front of Q term
		DATATYPE inv_sigma2_t2_scaled;	// 2 times sigma-squared divided by t_s squared
		DATATYPE tp_scaled;				// Dimensionless pulse width tp/t_scl
		DATATYPE alpha_m_scaled;		// Scaled absorption length
		DATATYPE alpha_r_scaled;		// Scaled reflective length

		// Heat Transfer Coefficients
		DATATYPE alpha_s;				// Heat transfer coefficient
		DATATYPE alpha_E;				// Heat transfer coefficient East (x=L, z>0)
		DATATYPE alpha_W;				// Heat transfer coefficient West (x=0, z>0)
		DATATYPE alpha_N;				// Heat transfer coefficient East (y=L, z>0)
		DATATYPE alpha_south;			// Heat transfer coefficient West (y=0, z>0)		Note to change name later
		DATATYPE alpha_SE;				// Heat transfer coefficient Substrate, East (x=L,z<0)
		DATATYPE alpha_SW; 				// Heat transfer coefficient Substrate, West (x=0, z<0)
		DATATYPE alpha_SN;				// Heat transfer coefficient Substrate, North (y=L,z<0)
		DATATYPE alpha_SS; 				// Heat transfer coefficient Substrate, South (y=0, z<0)

		// Biot Numbers
		DATATYPE Bi;					// Biot number for substrate bottom (south)
		DATATYPE Bi_E;					// Biot number at x=L in film z>0
		DATATYPE Bi_W;					// Biot number at x=0 in film z>0
		DATATYPE Bi_N;					// Biot number at y=L in film z>0
		DATATYPE Bi_S;					// Biot number at y=0 in film z>0
		DATATYPE Bi_SE;					// Biot number at x=L in substrate z<0
		DATATYPE Bi_SW;					// Biot number at x=0 in substrate z<0
		DATATYPE Bi_SN;					// Biot number at y=L in substrate z<0
		DATATYPE Bi_SS;					// Biot number at y=0 in substrate z<0

		DATATYPE Bi_E_star;
		DATATYPE Bi_W_star;
		DATATYPE Bi_N_star;
		DATATYPE Bi_S_star;
		DATATYPE Bi_SE_star;
		DATATYPE Bi_SW_star;
		DATATYPE Bi_SN_star;
		DATATYPE Bi_SS_star;

		DATATYPE D_N, D_S, D_E, D_W, D_SE, D_SW, D_SN, D_SS;			// First coefficient of ghost point
		DATATYPE E_N, E_S, E_E, E_W, E_SE, E_SW, E_SN, E_SS;			// Second coefficient of ghost point
		//== These coefficients are declared in gadit_solver.h

		// Dimensionless Parameters
		DATATYPE K1;					// Film Diffusitivity							(Dimensionless)
		DATATYPE K2;					// Substrate Diffusitivity						(Dimensionless)

		DATATYPE K1_t_kratio_db2;			// K1 times k_ratio divided by 2

		// Derived parameters common to each spacial point,
		// precomputed on CPU to remove redundant GPU computations.
		// Only required if expression for nonlinear functions
		// are complicated but constant valued terms.
		DATATYPE inv_w;
		DATATYPE beta2;
		DATATYPE w2;
		DATATYPE two_w;
		DATATYPE three_b;
		DATATYPE scaled_cN;
		DATATYPE cK_b2;
		DATATYPE scaled_cK_b3;
		DATATYPE temp_tol;
		bool film_evolution;
		bool melting_switch;
		bool temp_adapt;
		bool output_sub_temp;
		bool constant_parameters;
		bool time_varying_surf_tens;
		bool time_varying_visc;
		bool spat_varying_visc;
		bool nonlin_sub_heat;
		bool nonlin_grid;
		bool sub_in_plane_diff;
		int TC_model;
		DATATYPE T_eps_newt;
		int n_iter_max;

		// Required function. Can leave empty, but definition required.
		void compute_derived_parameters()
		{
			inv_w = 1.0 / w;
			beta2 = beta*beta;
			w2 = w*w;
			three_b = 3.0 * b;
			two_w = 2.0 * w;
			scaled_cN = 0.25*cN*inv_w;
			cK_b2 = cK*b*b;
			scaled_cK_b3 = 3.0*cK*b*b*b;
			// Thermal parameters
			l_scl2 = l_scl*l_scl;
			t_scl = 3.0*visc_0*l_scl/gamma_0;
			T_scl = t_scl*energy_0/alpha_m_inv/rho_m/Ceff_m/tp;
			K1 = k_m/(rho_m*Ceff_m*l_scl2)*t_scl;
			K2 = k_sio2/rho_sio2/Ceff_sio2*t_scl/(l_scl2);
			k_ratio = k_sio2 / k_m;      //k_m/k_sio2;
			gamma_ratio = T_scl * gamma_T / gamma_0;
			Temp_melt = 1358 / T_scl;
			sigma = tp / 2.0 / sqrt(2.0*log(2.0));
			Bi = 0.1;//alpha_s * l_scl / k_sio2;
			Temp_amb = Temp_room / T_scl;
			cap_M = (vdw_n - vdw_m) / ((vdw_n - 1.0)*(vdw_m - 1.0));
			//kappa_dp = gamma_0 * (1 - cos(theta_c / 180.0*atan(1.0)*4.0)) / (cap_M*b*l_scl);
			kappa_dp = gamma_0 * (tan(theta_c / 180.0*M_PI)*tan(theta_c / 180.0*M_PI)) / (2.0*cap_M*b*l_scl);
			Q_coef = tp*zeta / sqrt(2.0*(M_PI*1.0)) / sigma / (l_scl)*alpha_m_inv;
			inv_sigma2_t2_scaled = 1.0 / (2.0 * pow(sigma/t_scl, 2));
			tp_scaled = tp/t_scl;
			alpha_m_scaled = 1.0 / alpha_m_inv * l_scl;
			alpha_r_scaled = 1.0 / alpha_r_inv * l_scl;
			K1_t_kratio_db2 = K1 * k_ratio / 2.0;
			Bi_E = alpha_E * l_scl / k_m;
			Bi_W = alpha_W * l_scl / k_m;
			Bi_N = alpha_N * l_scl / k_m;
			Bi_S = alpha_south * l_scl / k_m;

			Bi_SN = alpha_SN * l_scl / k_sio2;
			Bi_SS = alpha_SS * l_scl / k_sio2;
			Bi_SE = alpha_SE * l_scl / k_sio2;
			Bi_SW = alpha_SW * l_scl / k_sio2;

			//== Define coefficients for approximation of Robin boundary condition e.g. T_0 = D T_1 + E
			//== Using cardinal directions (N=north 
			//								y
			//			N					^
			//			:					:
			//		 W-----E				:-----> x
			//			:	
			//			S
		}

		// String data to print parameter values to file.
		// Required function. Can leave empty, but definition required.
		std::string to_string()
		{
			std::string output;
			output = format_parameter_output::make_title("Thermal Model");

			output += "cK   = " + format_parameter_output::datatype(this->cK) + "\n";
			output += "cC   = " + format_parameter_output::datatype(this->cC) + "\n";
			output += "cN   = " + format_parameter_output::datatype(this->cN) + "\n";
			output += "beta = " + format_parameter_output::datatype(this->beta) + "\n";
			output += "b    = " + format_parameter_output::datatype(this->b) + "\n";
			output += "w    = " + format_parameter_output::datatype(this->w) + "\n";
			output += "T_scl    = " + format_parameter_output::datatype(this->T_scl) + "\n";
			output += "t_scl    = " + format_parameter_output::datatype(this->t_scl) + "\n";
			output += "K_1    = " + format_parameter_output::datatype(this->K1) + "\n";
			output += "K_2    = " + format_parameter_output::datatype(this->K2) + "\n";
			output += "Bi    = " + format_parameter_output::datatype(this->Bi) + "\n";
			output += "\n";

			return output;
		}


		// Optional command
		DATATYPE getGrowthRate(DATATYPE h, DATATYPE q)
		{
			DATATYPE gRate;
			DATATYPE f1;
			DATATYPE f2;

			DATATYPE h2 = h*h;
			DATATYPE h3 = h2*h;

			DATATYPE tau = tanh((h - 2 * b)*inv_w);
			DATATYPE kappa = tau - 1;;

			DATATYPE eta = h2 + beta*beta;
			DATATYPE eta2 = eta*eta;
			DATATYPE invEta = 1.0 / eta;

			DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
			DATATYPE invH = 1.0 / h;

			f2 = -h*shareNematic*(h2*two_w - w*eta + h*eta*(tau - 1));
			f2 += cK_b2*(2 - three_b*invH);

			f1 = cC*h3;

			DATATYPE q2 = q*q;
			gRate = -(f1*q2 - f2)*q2;

			return gRate;


		}

		// Optional command
		DATATYPE getMaxGrowthMode(DATATYPE h)
		{
			DATATYPE qm;
			DATATYPE f1;
			DATATYPE f2;
			DATATYPE f3;

			DATATYPE h2 = h*h;
			DATATYPE h3 = h2*h;

			DATATYPE tau = tanh((h - 2 * b)*inv_w);
			DATATYPE kappa = tau - 1;;

			DATATYPE eta = h2 + beta*beta;
			DATATYPE invEta = 1.0 / eta;

			DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
			DATATYPE invH = 1.0 / h;

			f2 = -h*shareNematic*(h2*two_w - w*eta + h*eta*(tau - 1));
			f2 += cK_b2*(2 - three_b*invH);

			f1 = cC*h3;

			f3 = cD*h2;	//Note that we need to define cD;

			qm = sqrt(f2 / (2.0*f1));

			return qm;
		}
	};

#endif
