// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2011 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// check serialization for Triangulation<1,dim>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include "serialization.h"

namespace dealii
{
  template <int dim, int spacedim>
  bool
  operator==(const Triangulation<dim, spacedim> &t1,
             const Triangulation<dim, spacedim> &t2)
  {
    // test a few attributes, though we can't
    // test everything unfortunately...
    if (t1.n_active_cells() != t2.n_active_cells())
      return false;

    if (t1.n_cells() != t2.n_cells())
      return false;

    if (t1.n_faces() != t2.n_faces())
      return false;

    typename Triangulation<dim, spacedim>::cell_iterator c1 = t1.begin(),
                                                         c2 = t2.begin();
    for (; (c1 != t1.end()) && (c2 != t2.end()); ++c1, ++c2)
      {
        for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
          {
            if (c1->vertex(v) != c2->vertex(v))
              return false;
            if (c1->vertex_index(v) != c2->vertex_index(v))
              return false;
          }

        for (const unsigned int f : GeometryInfo<dim>::face_indices())
          {
            if (c1->face(f)->at_boundary() != c2->face(f)->at_boundary())
              return false;

            if (c1->face(f)->manifold_id() != c2->face(f)->manifold_id())
              return false;

            if (c1->face(f)->at_boundary())
              {
                if (c1->face(f)->boundary_id() != c2->face(f)->boundary_id())
                  return false;
              }
            else
              {
                if (c1->neighbor(f)->level() != c2->neighbor(f)->level())
                  return false;
                if (c1->neighbor(f)->index() != c2->neighbor(f)->index())
                  return false;
              }
          }

        if (c1->is_active() && c2->is_active() &&
            (c1->subdomain_id() != c2->subdomain_id()))
          return false;

        if (c1->level_subdomain_id() != c2->level_subdomain_id())
          return false;

        if (c1->material_id() != c2->material_id())
          return false;

        if (c1->user_index() != c2->user_index())
          return false;

        if (c1->user_flag_set() != c2->user_flag_set())
          return false;

        if (c1->manifold_id() != c2->manifold_id())
          return false;

        if (c1->is_active() && c2->is_active())
          if (c1->active_cell_index() != c2->active_cell_index())
            return false;

        if (c1->level() > 0)
          if (c1->parent_index() != c2->parent_index())
            return false;
      }

    // also check the order of raw iterators as they contain
    // something about the history of the triangulation
    typename Triangulation<dim, spacedim>::cell_iterator r1 = t1.begin(),
                                                         r2 = t2.begin();
    for (; (r1 != t1.end()) && (r2 != t2.end()); ++r1, ++r2)
      {
        if (r1->level() != r2->level())
          return false;
        if (r1->index() != r2->index())
          return false;
      }

    return true;
  }
} // namespace dealii


template <int dim, int spacedim>
void
do_boundary(Triangulation<dim, spacedim> &t1)
{
  typename Triangulation<dim, spacedim>::cell_iterator c1 = t1.begin();
  for (; c1 != t1.end(); ++c1)
    for (const unsigned int f : GeometryInfo<dim>::face_indices())
      if (c1->at_boundary(f))
        {
          c1->face(f)->set_boundary_id(42);
          //        c1->face(f)->set_manifold_id (43);
        }
}


template <int spacedim>
void
do_boundary(Triangulation<1, spacedim> &)
{}


template <int dim, int spacedim>
void
test()
{
  Triangulation<dim, spacedim> tria_1, tria_2;

  GridGenerator::hyper_cube(tria_1);
  tria_1.refine_global(2);
  tria_1.begin_active()->set_subdomain_id(1);
  tria_1.begin_active()->set_level_subdomain_id(4);
  tria_1.begin_active()->set_material_id(2);
  tria_1.begin_active()->set_user_index(3);
  tria_1.begin_active()->set_user_flag();
  tria_1.begin_active()->set_refine_flag(RefinementCase<dim>::cut_x);

  do_boundary(tria_1);

  verify(tria_1, tria_2);
}


int
main()
{
  initlog();
  deallog << std::setprecision(3);

  test<1, 1>();
  test<1, 2>();
  test<2, 2>();
  test<2, 3>();
  test<3, 3>();

  deallog << "OK" << std::endl;
}
