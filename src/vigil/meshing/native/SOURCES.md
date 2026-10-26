# FLATTEN MESH: reference planar solvers

`planar_solver.cpp` calls the published implementations rather than replacing their
objectives or optimisation algorithms. Vendor files are unmodified (copied from the
test_legs viewer, where these methods were compared).

- **libigl 2.5.0**, commit `fdaac01bcc52888994f7afd029dcc045dd408484`:
  <https://github.com/libigl/libigl/tree/v2.5.0>. Only the transitive header/source
  dependencies of ARAP and SCAF are needed. License: `vendor/libigl/LICENSE.MPL2`.
- **Triangle**, commit `62f02db9ab4ff4b62d5ff82a77c8ea458c84c23a`:
  <https://github.com/libigl/triangle>. Original notices and redistribution terms
  are in `vendor/triangle/README` and the source headers.
- **ARAP**: Liu et al., *A Local/Global Approach to Mesh Parameterization* (2008).
  <https://cs.harvard.edu/~sjg/papers/arap.pdf>. Uses libigl's per-element 2D ARAP
  with a single vertex pinned only to remove translation freedom. No dynamic
  regularisation or additional energy term. The boundary shape remains free.
  Initial circle area equals source surface area.
- **SCAF**: Jiang et al., *Simplicial Complex Augmentation Framework for Bijective
  Maps* (2017). <https://cims.nyu.edu/gcl/papers/Scaffold-2017.pdf>.
  The authors link libigl's 2D implementation:
  <https://github.com/jiangzhongshi/Scaffold-Map#libigl-version>.
  Calls `igl::triangle::scaf_precompute` / `scaf_solve` with symmetric Dirichlet
  energy, no positional constraints, and the authors' area-matched harmonic
  initialisation. Inputs are normalised to unit surface area for libigl's
  energy-minus-four convention and UVs are then restored to metres. Iteration stops
  at energy below `1e-12` to avoid a negative scaffold weight from round-off at isometry.

Built automatically on the first FLATTEN (CMake, a C++17 compiler and Eigen3 required):

```sh
cmake -S src/vigil/meshing/native -B src/vigil/meshing/native/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/vigil/meshing/native/build -j2
```

No result is saved until `../flatten.py` has validated finite coordinates, positive
triangle areas, a simple boundary and the area check. Invalid output raises an error;
another method is never substituted.
