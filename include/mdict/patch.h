#ifndef PATCH_H
#define PATCH_H

#include "include.h"
#include "che.h"

#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

/// Mesh dictionary learning and sparse coding namespace
namespace mdict {

class dictionary;

typedef vector<pair<index_t, index_t> > vpatches_t;

/// 
class patch
{
	public:
		vector<index_t> vertices;		///< Vertices of the patch.
		mat T;							///< Transformation matrix.
		vec x;							///< Center point.
		mat xyz;						///< Matrix of points.
		mat phi;
	
	public:
		static size_t expected_nv;		///< Expected number of patch vertices.

	public:
		patch() = default;
		~patch() = default;
		
		void init(	che * mesh,						///< input mesh.
					const index_t & v,				///< center vertex of the patch.
					const size_t & n_toplevels,		///< number of toplevels to jet fitting.
					const distance_t & radio,		///< euclidean radio in XY of the patch.
					index_t * _toplevel = NULL		///< aux memory to gather toplevel vertices.
					);

		void transform();
		
		void itransform();
		
		void reset_xyz(	che * mesh,
						vector<vpatches_t> & vpatches,
						const index_t & p,
						const index_t & threshold = NIL
						);

	private:
		/// Gather the vertices needed to compute the jet_fit_directions of the patch.
		void gather_vertices(	che * mesh,
								const index_t & v,
								const size_t & n_toplevels,
								index_t * toplevel
								);
		
		/// Gather the vertices filter by radio in the local coordinates require initialize T and x.
		void gather_vertices(	che * mesh,
								const index_t & v,
								const distance_t & radio,
								index_t * toplevel
								);
		
		/// Initialize transformation matrix T and translation vector x, using CGAL jet_fitting.
		void jet_fit_directions(che * mesh,
								const index_t & v
								);
		

	friend class dictionary;
};

} // mdict

#endif // PATCH_H
