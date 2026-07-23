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


#include <deal.II/fe/fe_face_nedelec.h>
#include <deal.II/fe/mapping_q1.h>

#include <string>

#include "../tests.h"

#include "shapes.h"

#define PRECISION 8


template <int dim>
void
plot_FE_FaceNedelec_shape_functions()
{
  MappingQ<dim>       m(1);
  FE_FaceNedelec<dim> p0(0);
  test_compute_functions(m, p0, "FaceNedelec0");
  FE_FaceNedelec<dim> p1(1);
  test_compute_functions(m, p1, "FaceNedelec1");
}


int
main()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(PRECISION) << std::fixed;
  deallog.attach(logfile);
  deallog << "FE_FaceNedelec<2>" << std::endl;
  plot_FE_FaceNedelec_shape_functions<2>();
  deallog << "FE_FaceNedelec<3>" << std::endl;
  plot_FE_FaceNedelec_shape_functions<3>();

  return 0;
}
