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
	if(nargs < 4)
	{
		printf("./run_geodesics <inputfile> <outputfile> <method>\n");
		printf("  <inputfile>  :  mesh file in format OFF or OBJ.\n");
		printf("  <outputfile> :  output file location, will be in text format.\n");
		printf("  <method> :  1=ptp_cpu, 2=heatflow_cpu, 3=ptp_gpu, 4=heatflow_gpu.\n");

		return;
	}

	const char * data_path = args[1];
	const char * outputfile = args[2];
	int method = atoi(args[3]);

	int verbosity = 1;
	int is_test = 0;		// If set, only 100 vertices are really computed. Used to timing different methods.
	if(is_test == 1) {
		printf("WARNING: Running in test mode, will only compute mean distance values for the first few vertices.\n");
	}

	cout << "Handling input file '" << data_path << "', using method " << method << "\n";
	cout << "Will write to output file '" << outputfile << "'.\n";

	che * mesh;
	mesh = new che_off(data_path);
	size_t n_vertices = mesh->n_vertices();

	cout << "Mesh with " << n_vertices << " vertices loaded.\n";
	distance_t * mean_dists = new distance_t[mesh->n_vertices()];	// mean distance from one vertex to all others. kept over all iterations, one value added in each iteration.

	index_t verts_to_compute = n_vertices;
	if(is_test == 1) {
		verts_to_compute = 100;
	}

	for(index_t source_vert = 0; source_vert < verts_to_compute; source_vert++) {
		vector <index_t> source = { source_vert };

		index_t * toplesets = new index_t[n_vertices];
		index_t * sorted_index = new index_t[n_vertices];
		vector<index_t> limits;
  	//	cout << "Computing toplesets\n";

		double st;
		distance_t * dist;

		if(method == 1 || method == 3) {
			//mesh = new che_off(data_path);
			mesh->compute_toplesets(toplesets, sorted_index, limits, source);
		}



		if(method == 1) {
			//cout << "Running ptp_cpu method\n";
			/* run_ptp_cpu(mesh, source, {limits, sorted_index}); */
			dist = new distance_t[mesh->n_vertices()];	// geodesic distances of the source vertex to all other vertices. reset in each iteration.
			const toplesets_t & toplesets2 = {limits, sorted_index};
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
			 const toplesets_t & toplesets2 = {limits, sorted_index};
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

		//delete mesh;
		delete [] toplesets;
		delete [] sorted_index;
    if (dist) delete [] dist;

		if(verbosity > 0) {
			if(source_vert % 100 == 0) {
			    printf("  At source vertex %d.\n", (int)source_vert);
			}
			else if (source_vert % 1000 == 0) {
			    printf("At source vertex %d.\n", (int)source_vert);
	    }
	  }


	}

	if(is_test == 1) {
		printf("Mean geodesic distances for first 3 verts were: %f %f %f.\n", mean_dists[0], mean_dists[1], mean_dists[2]);
	}
	save_dists(outputfile, mean_dists, n_vertices);
  printf("Saved mean distances to file '%s'.\n", outputfile);

}


void save_dists(const char * outfile, distance_t * mean_dists, size_t num_dists) {
	int save_method = 2;   // 1=binary, 2=CSV
	if(save_method == 1) {
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
 if(save_method == 2) {
	 fstream f;
    f.open(outfile, ios::out);
    for(size_t i=0; i < num_dists; i++) {
			if(i < (num_dists-1)) {
    		f << mean_dists[i] << ';';
			} else {
				f << mean_dists[i];
			}
		}
    f.close();
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
