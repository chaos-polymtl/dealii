// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2002 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <string>

#include "../tests.h"

template <int dim>
void
test1()
{
  Triangulation<dim> tria;
  GridIn<dim>        gi;
  gi.attach_triangulation(tria);
  std::ifstream in(SOURCE_DIR "/grid_in/2d.inp");
  gi.read_ucd(in);

  GridOut grid_out;
  grid_out.set_flags(GridOutFlags::Ucd(true));
  grid_out.write_ucd(tria, deallog.get_file_stream());
}


template <int dim>
void
test2()
{
  // read a much larger grid (30k
  // cells). with the old grid
  // reordering scheme, this took >90
  // minutes (exact timing not
  // available, program was killed
  // before), with the new one it
  // takes less than 8 seconds
  //
  // note that the input file is no good: it
  // contains two pairs of two cells, where
  // the two members of each pair share 3
  // vertices (in 2d) -- this can of course
  // not work properly. it makes the
  // grid_in_02 testcase fail when using this
  // input file, but grid_in_02/2d.xda is a
  // corrected input file.
  Triangulation<dim> tria(Triangulation<dim>::none, true);
  GridIn<dim>        gi;
  gi.attach_triangulation(tria);
  std::ifstream in(SOURCE_DIR "/grid_in/2d.xda");
  try
    {
      gi.read_xda(in);
    }
  catch (typename Triangulation<dim>::DistortedCellList &dcv)
    {
      // ignore the exception that we
      // get because the mesh has
      // distorted cells
      deallog << dcv.distorted_cells.size() << " cells are distorted."
              << std::endl;
    }


  int hash  = 0;
  int index = 0;
  for (typename Triangulation<dim>::active_cell_iterator c =
         tria.begin_active();
       c != tria.end();
       ++c, ++index)
    for (const unsigned int i : c->vertex_indices())
      hash += (index * i * c->vertex_index(i)) % (tria.n_active_cells() + 1);
  deallog << hash << std::endl;
}


template <int dim>
void
test3()
{
  Triangulation<dim> tria;
  GridIn<dim>        gi;
  gi.attach_triangulation(tria);
  gi.read(SOURCE_DIR "/grid_in/2d.nc");

  GridOut       grid_out;
  std::ofstream gnufile("grid_in_2d.gnuplot");
  grid_out.write_gnuplot(tria, gnufile);
}


template <int dim>
void
check_file(const std::string name, typename GridIn<dim>::Format format)
{
  Triangulation<dim> tria(Triangulation<dim>::none, true);
  GridIn<dim>        gi;
  gi.attach_triangulation(tria);
  try
    {
      gi.read(name, format);
    }
  catch (typename Triangulation<dim>::DistortedCellList &dcv)
    {
      // ignore the exception
      deallog << dcv.distorted_cells.size() << " cells are distorted."
              << std::endl;
    }

  deallog << '\t' << tria.n_vertices() << '\t' << tria.n_cells() << std::endl;
}

void
filename_resolution()
{
  check_file<2>(std::string(SOURCE_DIR "/grid_in/2d.inp"), GridIn<2>::ucd);
  check_file<2>(std::string(SOURCE_DIR "/grid_in/2d.xda"), GridIn<2>::xda);
}


int
main()
{
  initlog();
  deallog.get_file_stream() << std::setprecision(2);

  test1<2>();
  test2<2>();

  filename_resolution();
}
