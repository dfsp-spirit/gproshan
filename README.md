# gproshan
### geometry processing and shape analysis framework

![](gproshan.png) 


## Description
This framework includes some algorithms of Geometry Processing and Shape Analysis as part of our graduate research.


## Build and Run
Install all dependencies and run:

	make

finally execute:

	./gproshan [input mesh paths]

### Dependencies (linux)
g++ 7.2, cuda >= 9.1, libarmadillo, libeigen, libsuitesparse, libopenblas, opengl, gnuplot, libcgal

## Contributions

### CHE implementation
We have implemented a [Compact Half-Edge (CHE)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.523.7580) data structure to manipulated the meshes.
See the paper: [CHE: A scalable topological data structure for triangular meshes](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.523.7580) for more details.

### Geodesics
We proposed a CPU/GPU parallel algorithm to compute geodesics distances on triangular meshes. Our
approach is competitive with the current methods and is simple to implement. Please cite our paper:

[An Iterative Parallel Algorithm for Computing Geodesic Distances on Triangular Meshes]()

```bibtex
@article{ptp2018,
	author	= { Luciano A. Romero Calla and Lizeth J. Fuentes Perez and Anselmo A. Montenegro and Marcos Lage },
	title	= { An Iterative Parallel Algorithm for Computing Geodesic Distances on Triangular Meshes },
	year	= { 2018 },
	url	= { }
}
```

Also, we have implemented the [Fast Marching algorithm](), and the [Heat method](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/index.html).

### Dictionary Learning
We proposed a Dictionary Learning and Sparse Coding framework,  to solve the problems of Denoising, Inpainting, and Multiresolution on triangular meshes. This work is still in process. Please  cite our work:

[A Dictionary Learning-based framework on Triangular Meshes]()

```bibtex
@article{dlspf2018,
	author	= { Luciano A. Romero Calla and Lizeth J. Fuentes Perez and Anselmo A. Montenegro },
	title	= { A Dictionary Learning-based framework on Triangular Meshes },
	year	= { 2018 },
	url	= { }
}
```

### Hole repairing
We implemented repairing mesh holes in two steps:

1. Generate a mesh to cover the hole (modified algorithm base on ...).
2. Approximate the curvature solving the Poisson equation and using Biharmonic splines.

### Decimation
We are implementing the algorithm described by the paper [Stellar Mesh Simplification Using Probabilistic Optimization](https://doi.org/10.1111/j.1467-8659.2004.00811.x),
to compute a mesh simplification.

### Fairing
We implemented Spectral and Taubin algorithms to smooth a mesh surface. See the Chapter 4 of the book
[Polygon Mesh Processing](http://www.pmp-book.org/).

### Laplacian and signatures
Laplace-Beltrami operator and its eigen decomposition, WKS, HKS, GPS signatures.

## Documentation
Execute:

	doxygen Doxyfile

to generate the documentation in html and latex.

## Viewer
The viewer was initially based in the viewer of [https://github.com/dgpdec/course](https://github.com/dgpdec/course). The current viewer uses VAO and VBO to render, and the shaders have been modified and upgraded.

## Authors
- [Lizeth Joseline Fuentes Pérez](https://github.com/lishh)
- [Luciano Arnaldo Romero Calla](https://github.com/larc)

