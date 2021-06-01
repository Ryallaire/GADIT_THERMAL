CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc -ccbin /usr/bin
RM=/bin/rm -f


# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

OBJS = main.cu.o
DEPS = gadit_solver.h solver_template.h initial_condition_list.h solver_template.h newton_iterative_method.h backup_manager.h timestep_manager.h status_logger.h output.h subf_temp.h
DEPS += work_space.h memory_manager.h parameters.h output.h cuda_parameters.h dimensions.h 
DEPS += model_thermal1.h change_mat_par.h model_list.h
DEPS += nonlinear_functions.h nonlinear_penta_solver.h 
DEPS += tridiag_solver.h boundary_conditions.h compute_Jx.h compute_Jy_F.h

TARGET = exec
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA) -lstdc++fs

.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

#%.cu.o: %.cu $(DEPS)
%.cu.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	#$(NVCC) $(NVCCFLAGS) -c $@ -o $<

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.h.o: %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@


$(TARGET): $(OBJS) Makefile
	$(LINKLINE)

clean:
	${RM} *.o IC
# DO NOT DELETE

main.o: gadit_solver.h initial_condition_list.h work_space.h memory_manager.h
main.o: cuda_parameters.h dimensions.h parameters.h /usr/include/math.h
main.o: solver_template.h output.h newton_iterative_method.h
main.o: nonlinear_penta_solver.h compute_Jy_F.h boundary_conditions.h
main.o: compute_Jx.h model_list.h model_default.h model_nlc.h model_polymer.h
main.o: model_constant.h model_thermal1.h backup_manager.h timestep_manager.h
main.o: status_logger.h subf_temp.h /usr/include/stdio.h change_mat_par.h
