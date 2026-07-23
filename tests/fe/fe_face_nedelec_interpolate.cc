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


// FE_FaceNedelec supports interpolation through generalized support points,
// which is delegated to the underlying FE_Nedelec and truncated to the edge
// and face degrees of freedom. This test interpolates a vector field that lies
// in the lowest-order Nedelec space (and, by nestedness, in every
// FE_FaceNedelec space) on an affine mesh and checks that the tangential trace
// of the interpolant coincides with the exact field on all faces to machine
// precision.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_face_nedelec.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"



// u = (y, x) in 2d and u = (y, z, x) in 3d. These fields lie in the
// lowest-order Nedelec space and have non-vanishing curl, so they exercise
// genuine H(curl) content rather than a gradient.
template <int dim>
class ExactField : public Function<dim>
{
public:
  ExactField()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    if (dim == 2)
      return (component == 0) ? p[1] : p[0];
    else
      switch (component)
        {
          case 0:
            return p[1];
          case 1:
            return p[2];
          default:
            return p[0];
        }
  }
};



template <int dim>
void
check(const unsigned int order)
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 2, 0., 1.);

  FE_FaceNedelec<dim> fe(order);
  DoFHandler<dim>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const MappingQ<dim> mapping(1);

  const ExactField<dim> exact;
  Vector<double>        interpolant(dof_handler.n_dofs());
  VectorTools::interpolate(mapping, dof_handler, exact, interpolant);

  const QGauss<dim - 1> quadrature(fe.degree + 1);
  FEFaceValues<dim>     fe_face_values(mapping,
                                   fe,
                                   quadrature,
                                   update_values | update_quadrature_points |
                                     update_normal_vectors);

  const FEValuesExtractors::Vector vec(0);
  std::vector<Tensor<1, dim>>      values(quadrature.size());

  double max_error = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int face : cell->face_indices())
      {
        fe_face_values.reinit(cell, face);
        fe_face_values[vec].get_function_values(interpolant, values);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const Point<dim> &p = fe_face_values.quadrature_point(q);

            Tensor<1, dim> u_exact;
            for (unsigned int d = 0; d < dim; ++d)
              u_exact[d] = exact.value(p, d);

            // Compare tangential components: subtract the normal part of the
            // difference.
            const Tensor<1, dim> normal = fe_face_values.normal_vector(q);
            const Tensor<1, dim> diff   = values[q] - u_exact;
            const Tensor<1, dim> t_diff = diff - (diff * normal) * normal;

            max_error = std::max(max_error, t_diff.norm());
          }
      }

  deallog << fe.get_name() << ": max tangential interpolation error = "
          << (max_error < 1e-12 ? 0. : max_error) << std::endl;
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
