#ifndef RUN_GEODESICS_H
#define RUN_GEODESICS_H

#include "geodesics.h"
#include "geodesics_ptp.h"


// geometry processing and shape analysis framework
namespace gproshan {


/// Execute performance and accuracy test for ptp algorithm on cpu and gpu.
void run_geodesics(const int & nargs, const char ** args);

double run_fast_marching(che * mesh, const std::vector<index_t> & source);

double run_ptp_cpu(che * mesh, const std::vector<index_t> & source, const toplesets_t & toplesets);

double run_heat_method_cholmod(che * mesh, const std::vector<index_t> & source);


#ifdef GPROSHAN_CUDA

double run_ptp_gpu(che * mesh, const std::vector<index_t> & source, const toplesets_t & toplesets);

double run_heat_method_gpu(che * mesh, const std::vector<index_t> & source);



#endif // GPROSHAN_CUDA



} // namespace gproshan

#endif // RUN_GEODESICS_H

