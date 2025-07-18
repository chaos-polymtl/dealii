// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2008 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



//


#include "../tests.h"

// all include files you need here

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
// #include <deal.II/fe/fe_q.h>
#include <deal.II/base/observer_pointer.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <math.h>

#include <iostream>
#include <string>

std::ofstream logfile("output");


template <int dim>
class LaplaceKernelIntegration
{
public:
  LaplaceKernelIntegration();
  ~LaplaceKernelIntegration();

  void
  run();

  void
  compute_SD_integral_on_cell(
    std::vector<double>                                     &dst,
    typename DoFHandler<dim, dim + 1>::active_cell_iterator &cell,
    const Point<dim + 1>                                    &point);

private:
  double
  term_S(const Tensor<1, 3> &r,
         const Tensor<1, 3> &a1,
         const Tensor<1, 3> &a2,
         const Tensor<1, 3> &n,
         const double       &rn_c);

  double
  term_D(const Tensor<1, 3> &r, const Tensor<1, 3> &a1, const Tensor<1, 3> &a2);

  ObserverPointer<FEValues<dim, dim + 1>> fe_values;
};

template <>
LaplaceKernelIntegration<2>::LaplaceKernelIntegration()
{
  static FE_DGP<2, 3>   fe(0);
  std::vector<Point<2>> qps(5);
  qps[0] = Point<2>(0, 0);
  qps[1] = Point<2>(0, 1);
  qps[2] = Point<2>(1, 0);
  qps[3] = Point<2>(1, 1);
  qps[4] = Point<2>(.5, .5);
  std::vector<double>  ws(5, 1.);
  static Quadrature<2> quadrature(qps, ws);
  fe_values = new FEValues<2, 3>(
    fe, quadrature, update_values | update_jacobians | update_normal_vectors);
}

template <int dim>
LaplaceKernelIntegration<dim>::~LaplaceKernelIntegration()
{
  FEValues<dim, dim + 1> *fp = fe_values;
  fe_values                  = nullptr;
  delete fp;
}


template <>
void
LaplaceKernelIntegration<2>::compute_SD_integral_on_cell(
  std::vector<double>                    &dst,
  DoFHandler<2, 3>::active_cell_iterator &cell,
  const Point<3>                         &point)
{
  Assert(dst.size() == 2, ExcDimensionMismatch(dst.size(), 2));
  fe_values->reinit(cell);
  std::vector<DerivativeForm<1, 2, 3>> jacobians = fe_values->get_jacobians();
  std::vector<Tensor<1, 3>> normals = fe_values->get_normal_vectors();

  Tensor<1, 3> n, n_c;
  Tensor<1, 3> r_c = point - cell->center();
  n_c              = normals[4];

  double              rn_c = r_c * n_c;
  std::vector<double> i_S(4);
  std::vector<double> i_D(4);
  for (unsigned int q_point = 0; q_point < 4; ++q_point)
    {
      const Tensor<1, 3> r  = point - cell->vertex(q_point);
      const Tensor<1, 3> a1 = transpose(jacobians[q_point])[0];
      const Tensor<1, 3> a2 = transpose(jacobians[q_point])[1];
      n                     = normals[q_point];
      i_S[q_point]          = term_S(r, a1, a2, n, rn_c);
      i_D[q_point]          = term_D(r, a1, a2);
    }
  dst[0] = (i_S[3] - i_S[1] - i_S[2] + i_S[0]);
  dst[1] = (i_D[3] - i_D[1] - i_D[2] + i_D[0]);
}

template <int dim>
double
LaplaceKernelIntegration<dim>::term_S(const Tensor<1, 3> &r,
                                      const Tensor<1, 3> &a1,
                                      const Tensor<1, 3> &a2,
                                      const Tensor<1, 3> &n,
                                      const double       &rn_c)
{
  Tensor<1, 3> ra1 = cross_product_3d(r, a1);
  Tensor<1, 3> ra2 = cross_product_3d(r, a2);
  Tensor<1, 3> a12 = cross_product_3d(a1, a2);

  double integral = -1. / 2. / numbers::PI *
                    (-ra1 * n / a1.norm() * asinh(r * a1 / ra1.norm()) +
                     ra2 * n / a2.norm() * asinh(r * a2 / ra2.norm()) +
                     rn_c * atan2(ra1 * ra2, r.norm() * (r * a12)));

  return integral;
}

template <int dim>
double
LaplaceKernelIntegration<dim>::term_D(const Tensor<1, 3> &r,
                                      const Tensor<1, 3> &a1,
                                      const Tensor<1, 3> &a2)
{
  Tensor<1, 3> ra1 = cross_product_3d(r, a1);
  Tensor<1, 3> ra2 = cross_product_3d(r, a2);
  Tensor<1, 3> a12 = cross_product_3d(a1, a2);

  double integral =
    1. / 2. / numbers::PI * atan2(ra1 * ra2, (r.norm() * (r * a12)));

  return integral;
}


double
integration(Point<3> point)
{
  Triangulation<2, 3> square;
  GridGenerator::hyper_cube<2, 3>(square, 0, 2);
  DoFHandler<2, 3>          dof_handler(square);
  static const FE_DGP<2, 3> fe(0);
  dof_handler.distribute_dofs(fe);

  DoFHandler<2, 3>::active_cell_iterator cell = dof_handler.begin_active();

  LaplaceKernelIntegration<2> laplace;
  std::vector<double>         integrals(2);
  laplace.compute_SD_integral_on_cell(integrals, cell, point);
  return integrals[0];
}



int
main()
{
  deallog.attach(logfile);
  deallog << std::fixed;
  deallog << std::setprecision(5);

  Point<3> point(.5, .5, 0);
  double   true_result = -3.163145629 / numbers::PI;
  deallog << "Error on  " << point << " : " << integration(point) - true_result
          << std::endl;

  point       = Point<3>(3, 3, 0);
  true_result = -.2306783616;
  deallog << "Error on  " << point << " : " << integration(point) - true_result
          << std::endl;

  point       = Point<3>(1.5, .5, 0);
  true_result = -1.006860525;
  deallog << "Error on  " << point << " : " << integration(point) - true_result
          << std::endl;

  return 0;
}
