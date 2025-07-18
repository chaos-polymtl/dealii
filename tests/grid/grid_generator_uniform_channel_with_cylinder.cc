// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2013 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

// Test output for GridGenerator::hyper_cross()

#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"



void
dim_2(std::ostream &os)
{
  const unsigned int d = 2;
  Triangulation<d>   tr;

  GridGenerator::uniform_channel_with_cylinder(
    tr, {3, 5, 4, 4}, 1, 2, 0.75, 4, 2, true, true);

  GridOut gout;
  gout.write_vtk(tr, os);
}

void
dim_3(std::ostream &os)
{
  const unsigned int d = 3;
  Triangulation<d>   tr;
  GridGenerator::uniform_channel_with_cylinder(
    tr, {3, 5, 4, 4}, 1, 2, 0.75, 4, 2, true, true);
  GridOut gout;
  gout.write_vtk(tr, os);
}


int
main()
{
  initlog(true);
  std::ostream &logfile = deallog.get_file_stream();
  dim_2(logfile);
  dim_3(logfile);
}
