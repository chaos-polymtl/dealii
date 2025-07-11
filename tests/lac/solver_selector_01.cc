// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2010 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Test the SolverSelector class.

#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>

#include "../tests.h"

#include "../testmatrix.h"


template <typename MatrixType, typename VectorType>
void
check(const MatrixType &A, const VectorType &f)
{
  std::vector<std::string> names;
  names.push_back("cg");
  names.push_back("bicgstab");
  names.push_back("gmres");
  names.push_back("fgmres");

  ReductionControl                       cont1(100, 0., 1.e-4, false, true);
  SolverControl                          cont2(100, 1.e-7, false, true);
  SolverSelector<VectorType>             solver;
  PreconditionSSOR<SparseMatrix<double>> pre;
  pre.initialize(A);

  VectorType u;
  u.reinit(f);

  std::vector<std::string>::const_iterator name;

  solver.set_control(cont1);
  for (name = names.begin(); name != names.end(); ++name)
    {
      solver.select(*name);
      u = 0.;
      solver.solve(A, u, f, pre);
    }

  solver.set_control(cont2);
  for (name = names.begin(); name != names.end(); ++name)
    {
      solver.select(*name);
      u = 0.;
      solver.solve(A, u, f, pre);
    }
}


int
main()
{
  std::ofstream logfile("output");
  //  logfile.setf(std::ios::fixed);
  deallog << std::setprecision(4);
  deallog.attach(logfile);

  unsigned int size = 37;
  unsigned int dim  = (size - 1) * (size - 1);

  deallog << "Size " << size << " Unknowns " << dim << std::endl;

  // Make matrix
  FDMatrix        testproblem(size, size);
  SparsityPattern structure(dim, dim, 5);
  testproblem.five_point_structure(structure);
  structure.compress();
  SparseMatrix<double> A(structure);
  testproblem.five_point(A);
  Vector<double> f(dim);
  f = 1.;

  check(A, f);
}
