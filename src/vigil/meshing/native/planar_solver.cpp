// FLATTEN MESH: SCAF and ARAP from the unmodified libigl reference implementations.
// Algorithms and licences are recorded in SOURCES.md. Called by ../flatten.py:
//   planar_solver <scaf|arap> <mesh.txt> <uv.txt> <iterations>
#include <igl/triangle/scaf.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::runtime_error("Expected method, input, output, iterations.");
    std::string method(argv[1]);
    int iterations = std::stoi(argv[4]);
    if (iterations < 1 || iterations > 1000) throw std::runtime_error("Invalid iteration limit.");
    std::ifstream input(argv[2]);
    int nv, nf;
    if (!(input >> nv >> nf) || nv < 3 || nf < 1) throw std::runtime_error("Invalid mesh input.");
    Eigen::MatrixXd V(nv, 3), uv;
    Eigen::MatrixXi F(nf, 3);
    for (int i = 0; i < nv; ++i)
      for (int j = 0; j < 3; ++j)
        if (!(input >> V(i, j))) throw std::runtime_error("Truncated vertices.");
    for (int i = 0; i < nf; ++i)
      for (int j = 0; j < 3; ++j)
        if (!(input >> F(i, j)) || F(i, j) < 0 || F(i, j) >= nv)
          throw std::runtime_error("Invalid faces.");
    Eigen::VectorXi boundary;
    // SCAF's reference energy subtracts 4 assuming unit total input area.
    // Normalize for that convention, then restore the original physical units.
    double source_scale = 1;
    if (method == "scaf") {
      Eigen::VectorXd area;
      igl::doublearea(V, F, area);
      source_scale = std::sqrt(area.sum() / 2);
      V /= source_scale;
    }
    igl::boundary_loop(F, boundary);
    Eigen::MatrixXd boundary_uv;
    igl::map_vertices_to_circle(V, boundary, boundary_uv);
    Eigen::VectorXd twice_area;
    igl::doublearea(V, F, twice_area);
    // Match the SCAF authors' initialization: circle of equal surface area.
    boundary_uv *= std::sqrt(twice_area.sum() / (2 * igl::PI));
    if (boundary.size() == nv) {
      uv.resize(nv, 2);
      for (int i = 0; i < nv; ++i) uv.row(boundary(i)) = boundary_uv.row(i);
    } else {
      igl::harmonic(V, F, boundary, boundary_uv, 1, uv);
      if (igl::flipped_triangles(uv, F).size())
        igl::harmonic(F, boundary, boundary_uv, 1, uv);
    }
    if (!uv.allFinite() || igl::flipped_triangles(uv, F).size())
      throw std::runtime_error("Could not initialize an orientation-preserving disk map.");
    Eigen::VectorXi pinned(0);
    Eigen::MatrixXd targets(0, 2);
    if (method == "scaf") {
      igl::triangle::SCAFData data;
      igl::triangle::scaf_precompute(V, F, uv, data,
          igl::MappingEnergyType::SYMMETRIC_DIRICHLET, pinned, targets, 0);
      for (int i = 0; i < iterations; ++i) {
        uv = igl::triangle::scaf_solve(data, 1);
        if (!uv.allFinite()) throw std::runtime_error("SCAF produced non-finite coordinates.");
        std::cout << "PROGRESS " << i + 1 << ' ' << std::setprecision(17)
                  << data.energy << std::endl;
        // At numerical isometry the reference scaffold weight tends to zero.
        // Stop before round-off can make that nonnegative weight negative.
        if (data.energy < 1e-12) break;
      }
    } else if (method == "arap") {
      igl::ARAPData data;
      // Optimize the paper's static ARAP energy. A single vertex removes only
      // translation freedom; it imposes no length, angle or boundary shape.
      Eigen::VectorXi anchor(1); anchor << 0;
      Eigen::MatrixXd anchor_uv = uv.topRows(1);
      data.with_dynamics = false;
      data.energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
      data.max_iter = iterations;
      if (!igl::arap_precomputation(V, F, 2, anchor, data))
        throw std::runtime_error("ARAP precomputation failed.");
      if (!igl::arap_solve(anchor_uv, data, uv)) throw std::runtime_error("ARAP solve failed.");
      std::cout << "PROGRESS " << iterations << " 0" << std::endl;
    } else {
      throw std::runtime_error("Unknown method (expected scaf or arap).");
    }
    std::ofstream output(argv[3]);
    uv *= source_scale;
    output << std::setprecision(17) << uv.rows() << '\n' << uv << '\n';
    if (!output) throw std::runtime_error("Could not write mapping output.");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
