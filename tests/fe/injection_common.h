// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2007 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// common framework to check the following: if we interpolate from one finite
// element on a cell to a richer finite element on a finer cell, then it
// shouldn't matter whether we go to the richer FE first and then to the finer
// cells, or the other way around. test this

#include <deal.II/base/logstream.h>

#include <deal.II/fe/fe_abf.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_dgp_nonparametric.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_bubbles.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_q_hierarchical.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/full_matrix.h>

#include <fstream>
#include <vector>

#include "../tests.h"


template <int dim>
void
test();



template <int dim>
void
do_check(const FiniteElement<dim> &coarse_fe, const FiniteElement<dim> &fine_fe)
{
  FullMatrix<double> injection_1(fine_fe.dofs_per_cell,
                                 coarse_fe.dofs_per_cell);
  FullMatrix<double> injection_2(fine_fe.dofs_per_cell,
                                 coarse_fe.dofs_per_cell);

  for (unsigned int child_1 = 0;
       child_1 < GeometryInfo<dim>::max_children_per_cell;
       ++child_1)
    for (unsigned int child_2 = 0;
         child_2 < GeometryInfo<dim>::max_children_per_cell;
         ++child_2)
      {
        injection_1 = 0;
        injection_2 = 0;

        // check 1: first to finer fe, then
        // to finer cells
        {
          FullMatrix<double> tmp1(fine_fe.dofs_per_cell,
                                  coarse_fe.dofs_per_cell);
          FullMatrix<double> tmp2(fine_fe.dofs_per_cell,
                                  coarse_fe.dofs_per_cell);
          fine_fe.get_interpolation_matrix(coarse_fe, tmp1);
          fine_fe.get_prolongation_matrix(child_1).mmult(tmp2, tmp1);
          fine_fe.get_prolongation_matrix(child_2).mmult(injection_1, tmp2);
        }

        // check 2: first to finer cells,
        // then to finer FE
        {
          FullMatrix<double> tmp1(coarse_fe.dofs_per_cell,
                                  coarse_fe.dofs_per_cell);
          FullMatrix<double> tmp2(fine_fe.dofs_per_cell,
                                  coarse_fe.dofs_per_cell);

          coarse_fe.get_prolongation_matrix(child_2).mmult(
            tmp1, coarse_fe.get_prolongation_matrix(child_1));

          fine_fe.get_interpolation_matrix(coarse_fe, tmp2);
          tmp2.mmult(injection_2, tmp1);
        }

        // print one of the matrices. to
        // reduce output, do so only for the
        // some of the matrices
        if (child_1 ==
            ((child_2 + 1) % GeometryInfo<dim>::max_children_per_cell))
          for (unsigned int i = 0; i < fine_fe.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < coarse_fe.dofs_per_cell; ++j)
              deallog << i << ' ' << j << ' ' << injection_1(i, j) << std::endl;

        // make sure that the two matrices
        // are pretty much equal
        for (unsigned int i = 0; i < fine_fe.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < coarse_fe.dofs_per_cell; ++j)
            injection_2(i, j) -= injection_1(i, j);
        AssertThrow(injection_2.frobenius_norm() <=
                      1e-12 * injection_1.frobenius_norm(),
                    ExcInternalError());
      }
}



int
main()
{
  std::ofstream logfile(logname);
  deallog << std::setprecision(6);

  deallog.attach(logfile);
  deallog.depth_console(0);

  test<1>();
  test<2>();
  test<3>();
}
