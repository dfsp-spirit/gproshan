#include "run_geodesics.h"

#include "che_off.h"
#include "geodesics_ptp.h"
#include "heat_flow.h"

#include <cassert>
#include <iterator>
#include<iostream>
#include<fstream>

using namespace std;


// geometry processing and shape analysis framework
namespace gproshan {


void run_geodesics(const int & nargs, const char ** args)
{
	if(nargs < 3)
	{
		printf("./run_geodesics <inputfile> <outputfile>\n");
		printf("  <inputfile>  :  mesh file in format OFF or OBJ.\n");
		printf("  <outputfile> :  output file location, will be in csv format.\n");

		return;
	}

	const char * data_path = args[1];
	const char * outputfile = args[2];

	int method = 4;
	cout << "Handling input file '" << data_path << "', using method " << method << "\n";
	cout << "Will write to output file '" << outputfile << "'.\n";

	che * mesh;
	mesh = new che_off(data_path);
	size_t n_vertices = mesh->n_vertices();

	cout << "Mesh with " << n_vertices << " vertices loaded.\n";
	distance_t * mean_dists = new distance_t[mesh->n_vertices()];	// mean distance from one vertex to all others. kept over all iterations, one value added in each iteration.

	for(index_t source_vert = 0; source_vert < n_vertices; source_vert++) {
		vector <index_t> source = { source_vert };

		index_t * toplesets = new index_t[n_vertices];
		index_t * sorted_index = new index_t[n_vertices];
		vector<index_t> limits;
  	//	cout << "Computing toplesets\n";

		mesh = new che_off(data_path);
		mesh->compute_toplesets(toplesets, sorted_index, limits, source);


		double st;
		distance_t * dist;
    const toplesets_t & toplesets2 = {limits, sorted_index};

		if(method == 1) {
			cout << "Running ptp_cpu method\n";
			/* run_ptp_cpu(mesh, source, {limits, sorted_index}); */
			dist = new distance_t[mesh->n_vertices()];	// geodesic distances of the source vertex to all other vertices. reset in each iteration.
			parallel_toplesets_propagation_cpu(dist, mesh, source, toplesets2);
		}

		if(method == 2) {
			dist = nullptr;
			if(dist) delete [] dist;
			dist = heat_flow(mesh, source, st);
		}

#ifdef GPROSHAN_CUDA
		if(method == 3) {
		   dist = new distance_t[mesh->n_vertices()];
	     parallel_toplesets_propagation_coalescence_gpu(dist, mesh, source, toplesets2);

		}


		if(method == 4) {
		    dist = nullptr;
	      if(dist) delete [] dist;
	      dist = heat_flow_gpu(mesh, source, st);
		}
#else
#endif // GPROSHAN_CUDA

		// Compute and save mean distance
		double dist_sum = 0.0;
	  double dist_mean_this_vert;
	  for (size_t vidx = 0; vidx < n_vertices; vidx++) {
					//printf("Dist to vert %d is: %f\n", vidx, dist[vidx]);
		    dist_sum += dist[vidx];
	  }
		dist_mean_this_vert = dist_sum / (double)n_vertices;
//		printf("Mean dist is %f.\n", dist_mean);
	  mean_dists[source_vert] = dist_mean_this_vert;



	// FREE MEMORY

		delete mesh;
		delete [] toplesets;
		delete [] sorted_index;
    if (dist) delete [] dist;

		if(source_vert % 100 == 0) {
		    printf("  At source vertex %d.\n", (int)source_vert);
		}
		else if (source_vert % 1000 == 0) {
		    printf("At source vertex %d.\n", (int)source_vert);
                }


	}


}


void save_dists(const char * outfile, distance_t * mean_dists, size_t num_dists) {
	ofstream wf(outfile, ios::out | ios::binary);
   if(!wf) {
      cout << "Cannot open distance output file." << endl;
   }
   for(size_t i = 0; i < num_dists; i++)
      wf.write((char *) &mean_dists[i], sizeof(distance_t));
   wf.close();
   if(!wf.good()) {
      cout << "Error occurred while writing distances." << endl;
   }
}


double run_fast_marching(che * mesh, const vector<index_t> & source)
{

	geodesics fm(mesh, source, geodesics::FM);


	return 0.;
}

double run_ptp_cpu(che * mesh, const vector<index_t> & source, const toplesets_t & toplesets)
{

	distance_t * dist = new distance_t[mesh->n_vertices()];
	parallel_toplesets_propagation_cpu(dist, mesh, source, toplesets);


	delete [] dist;

	return 0.;
}

double run_heat_method_cholmod(che * mesh, const vector<index_t> & source)
{
	double st;

	distance_t * dist = nullptr;
	if(dist) delete [] dist;
	dist = heat_flow(mesh, source, st);

	delete [] dist;

	return 0.;
}


#ifdef GPROSHAN_CUDA

double run_ptp_gpu(che * mesh, const vector<index_t> & source, const toplesets_t & toplesets)
{

	distance_t * dist = new distance_t[mesh->n_vertices()];
	parallel_toplesets_propagation_coalescence_gpu(dist, mesh, source, toplesets);


	delete [] dist;

	return 0.;
}

double run_heat_method_gpu(che * mesh, const vector<index_t> & source)
{
	double st;
	distance_t * dist = nullptr;
	if(dist) delete [] dist;
	dist = heat_flow_gpu(mesh, source, st);



	delete [] dist;

	return 0.;
}

#endif // GPROSHAN_CUDA



} // namespace gproshan
