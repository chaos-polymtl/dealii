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


// Check the restriction and prolongation matrices of FE_FaceNedelec, which
// are the leading principal blocks of FE_Nedelec's. We interpolate a vector
// field that lies in the lowest-order Nedelec space (and thus in every
// FE_FaceNedelec space) on a single cell and on its isotropically refined
// children. Prolongating the coarse degrees of freedom to each child must
// reproduce the directly interpolated child degrees of freedom, and
// restricting the child degrees of freedom (summing over the children, as all
// degrees of freedom are restriction-additive) must reproduce the coarse
// ones.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_face_nedelec.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"



// u = (y, x) in 2d and u = (y, z, x) in 3d. These fields lie in the
// lowest-order Nedelec space and have non-vanishing curl.
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
  FE_FaceNedelec<dim> fe(order);
  const MappingQ<dim> mapping(1);
  const ExactField<dim> exact;

  // Degrees of freedom of the exact field on a single coarse cell.
  Triangulation<dim> coarse_tria;
  GridGenerator::hyper_cube(coarse_tria, 0., 1.);
  DoFHandler<dim> coarse_dof_handler(coarse_tria);
  coarse_dof_handler.distribute_dofs(fe);

  Vector<double> coarse_u(coarse_dof_handler.n_dofs());
  VectorTools::interpolate(mapping, coarse_dof_handler, exact, coarse_u);

  Vector<double> coarse_local(fe.n_dofs_per_cell());
  coarse_dof_handler.begin_active()->get_dof_values(coarse_u, coarse_local);

  // Degrees of freedom of the exact field on the refined cell.
  Triangulation<dim> fine_tria;
  GridGenerator::hyper_cube(fine_tria, 0., 1.);
  fine_tria.refine_global(1);
  DoFHandler<dim> fine_dof_handler(fine_tria);
  fine_dof_handler.distribute_dofs(fe);

  Vector<double> fine_u(fine_dof_handler.n_dofs());
  VectorTools::interpolate(mapping, fine_dof_handler, exact, fine_u);

  double max_prolongation_error = 0.;
  double max_restriction_error  = 0.;

  Vector<double> restricted(fe.n_dofs_per_cell());

  const auto coarse_cell = fine_dof_handler.begin(0);
  for (unsigned int child = 0; child < coarse_cell->n_children(); ++child)
    {
      Vector<double> child_direct(fe.n_dofs_per_cell());
      coarse_cell->child(child)->get_dof_values(fine_u, child_direct);

      // The field lies in the finite element space, so prolongating the
      // coarse degrees of freedom must reproduce the directly interpolated
      // child degrees of freedom.
      Vector<double> child_prolonged(fe.n_dofs_per_cell());
      fe.get_prolongation_matrix(child).vmult(child_prolonged, coarse_local);

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        max_prolongation_error =
          std::max(max_prolongation_error,
                   std::abs(child_prolonged(i) - child_direct(i)));

      // All degrees of freedom are restriction-additive, so the coarse
      // degrees of freedom are the sum of the children's contributions.
      Vector<double> contribution(fe.n_dofs_per_cell());
      fe.get_restriction_matrix(child).vmult(contribution, child_direct);
      restricted += contribution;
    }

  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    max_restriction_error =
      std::max(max_restriction_error,
               std::abs(restricted(i) - coarse_local(i)));

  // The prolongation matrices of FE_Nedelec are computed by a least-squares
  // procedure with a degree-dependent tolerance, so allow for a slightly
  // larger threshold than machine precision.
  deallog << fe.get_name() << ": max prolongation error = "
          << (max_prolongation_error < 1e-10 ? 0. : max_prolongation_error)
          << ", max restriction error = "
          << (max_restriction_error < 1e-10 ? 0. : max_restriction_error)
          << std::endl;
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
