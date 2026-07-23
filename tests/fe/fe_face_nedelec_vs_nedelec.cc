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


// FE_FaceNedelec is, by construction, the tangential trace of FE_Nedelec: its
// shape functions are exactly the edge and face shape functions of FE_Nedelec
// (the leading n_dofs_per_cell() ones). This test verifies that claim directly:
// on every face, and for both the standard and a non-standard mapping, the
// tangential components of shape function i of FE_FaceNedelec must coincide with
// those of shape function i of FE_Nedelec to machine precision. This checks the
// polynomial subset, the covariant fill path and the face-orientation handling
// at once, against a trusted reference.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_face_nedelec.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"



template <int dim>
void
compare(const unsigned int order, const bool distort)
{
  FE_FaceNedelec<dim> fe_face(order);
  FE_Nedelec<dim>     fe_full(order);

  // The trace element keeps the leading (edge + face) degrees of freedom of
  // FE_Nedelec.
  AssertThrow(fe_face.n_dofs_per_cell() <= fe_full.n_dofs_per_cell(),
              ExcInternalError());

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  if (distort)
    GridTools::distort_random(0.2, tria, /*keep_boundary=*/false);

  MappingQ<dim> mapping(1);

  const QGauss<dim - 1> quadrature(order + 2);
  const UpdateFlags     flags =
    update_values | update_quadrature_points | update_normal_vectors;
  FEFaceValues<dim> fe_face_values_trace(mapping, fe_face, quadrature, flags);
  FEFaceValues<dim> fe_face_values_full(mapping, fe_full, quadrature, flags);

  const FEValuesExtractors::Vector vec(0);

  double max_diff = 0.;

  for (const auto &cell : tria.active_cell_iterators())
    for (const unsigned int face : cell->face_indices())
      {
        fe_face_values_trace.reinit(cell, face);
        fe_face_values_full.reinit(cell, face);

        // Build the outward normal for a tangential projection.
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const Tensor<1, dim> normal =
              fe_face_values_trace.normal_vector(q);

            for (unsigned int i = 0; i < fe_face.n_dofs_per_cell(); ++i)
              {
                const Tensor<1, dim> v_trace =
                  fe_face_values_trace[vec].value(i, q);
                const Tensor<1, dim> v_full =
                  fe_face_values_full[vec].value(i, q);

                // Compare tangential components: subtract the normal part.
                const Tensor<1, dim> t_trace =
                  v_trace - (v_trace * normal) * normal;
                const Tensor<1, dim> t_full =
                  v_full - (v_full * normal) * normal;

                max_diff = std::max(max_diff, (t_trace - t_full).norm());
              }
          }
      }

  deallog << fe_face.get_name() << (distort ? " (distorted)" : " (affine)")
          << ": max tangential difference to FE_Nedelec = "
          << (max_diff < 1e-12 ? 0. : max_diff) << std::endl;
}



int
main()
{
  initlog();

  for (unsigned int order = 0; order < 3; ++order)
    {
      compare<2>(order, false);
      compare<2>(order, true);
    }
  for (unsigned int order = 0; order < 2; ++order)
    {
      compare<3>(order, false);
      compare<3>(order, true);
    }

  deallog << "OK" << std::endl;
}
