// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


// Check the hanging-node constraints of FE_FaceNedelec: on an adaptively
// refined mesh, fill a vector with random values, apply the constraints
// computed by DoFTools::make_hanging_node_constraints(), and verify that the
// resulting finite element function is tangentially continuous across all
// interior faces, in particular across the coarse-fine interfaces.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_face_nedelec.h>
#include <deal.II/fe/fe_interface_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"



template <int dim>
void
check(const unsigned int order)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(1);
  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  FE_FaceNedelec<dim> fe(order);
  DoFHandler<dim>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // A random vector made consistent by the hanging-node constraints is
  // tangentially continuous if and only if the constraints are correct.
  Vector<double> u(dof_handler.n_dofs());
  for (auto &v : u)
    v = random_value<double>();
  constraints.distribute(u);

  const QGauss<dim - 1>  quadrature(fe.degree + 1);
  FEInterfaceValues<dim> fe_interface_values(fe,
                                             quadrature,
                                             update_values |
                                               update_normal_vectors);

  const FEValuesExtractors::Vector vec(0);

  double       max_jump        = 0.;
  unsigned int n_hanging_faces = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int face : cell->face_indices())
      {
        if (cell->face(face)->at_boundary())
          continue;

        // Visit every interface exactly once: coarse-fine interfaces from
        // the finer side, equal-level interfaces from the cell with the
        // smaller id.
        if (cell->neighbor_is_coarser(face))
          {
            const std::pair<unsigned int, unsigned int> neighbor_face_subface =
              cell->neighbor_of_coarser_neighbor(face);
            fe_interface_values.reinit(cell,
                                       face,
                                       numbers::invalid_unsigned_int,
                                       cell->neighbor(face),
                                       neighbor_face_subface.first,
                                       neighbor_face_subface.second);
            ++n_hanging_faces;
          }
        else if (!cell->neighbor(face)->has_children() &&
                 cell->id() < cell->neighbor(face)->id())
          fe_interface_values.reinit(cell,
                                     face,
                                     numbers::invalid_unsigned_int,
                                     cell->neighbor(face),
                                     cell->neighbor_of_neighbor(face),
                                     numbers::invalid_unsigned_int);
        else
          continue;

        // Values of the constrained function on both sides of the interface.
        std::array<std::vector<Tensor<1, dim>>, 2> face_values;
        for (unsigned int i = 0; i < 2; ++i)
          {
            face_values[i].resize(quadrature.size());
            fe_interface_values.get_fe_face_values(i)[vec].get_function_values(
              u, face_values[i]);
          }

        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const Tensor<1, dim> normal = fe_interface_values.normal_vector(q);
            const Tensor<1, dim> diff = face_values[0][q] - face_values[1][q];
            const Tensor<1, dim> t_diff = diff - (diff * normal) * normal;

            max_jump = std::max(max_jump, t_diff.norm());
          }
      }

  // The constraints are built from FE_Nedelec's projection-based
  // interpolation matrices, so allow for a slightly larger threshold than
  // machine precision.
  deallog << fe.get_name() << ": " << n_hanging_faces
          << " hanging faces, max tangential jump = "
          << (max_jump < 1e-10 ? 0. : max_jump) << std::endl;
}



int
main()
{
  initlog();

  for (unsigned int order = 0; order < 3; ++order)
    check<2>(order);
  for (unsigned int order = 0; order < 2; ++order)
    check<3>(order);

  deallog << "OK" << std::endl;
}
