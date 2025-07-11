// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2012 - 2025 by the deal.II authors
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
// check collect_periodic_faces(b_id1, b_id2) for correct return values
//


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

/*
 * Generate a grid consisting of two disjoint cells, colorize the two
 * outermost faces. They will be matched via collect_periodic_faces
 *
 * The integer orientation determines the orientation of the second cell
 * (to get something else than the boring default orientation).
 */

/* The 2D case */
void
generate_grid(Triangulation<2> &triangulation, int orientation)
{
  Point<2> vertices_1[] = {
    Point<2>(-1., -3.),
    Point<2>(+1., -3.),
    Point<2>(-1., -1.),
    Point<2>(+1., -1.),
    Point<2>(-1., +1.),
    Point<2>(+1., +1.),
    Point<2>(-1., +3.),
    Point<2>(+1., +3.),
  };
  std::vector<Point<2>> vertices(&vertices_1[0], &vertices_1[8]);

  std::vector<CellData<2>> cells(2, CellData<2>());

  /* cell 0 */
  int cell_vertices_0[GeometryInfo<2>::vertices_per_cell] = {0, 1, 2, 3};

  /* cell 1 */
  int cell_vertices_1[2][GeometryInfo<2>::vertices_per_cell] = {
    {4, 5, 6, 7},
    {7, 6, 5, 4},
  };

  for (const unsigned int j : GeometryInfo<2>::vertex_indices())
    {
      cells[0].vertices[j] = cell_vertices_0[j];
      cells[1].vertices[j] = cell_vertices_1[orientation][j];
    }
  cells[0].material_id = 0;
  cells[1].material_id = 0;

  triangulation.create_triangulation(vertices, cells, SubCellData());

  Triangulation<2>::cell_iterator cell_1 = triangulation.begin();
  Triangulation<2>::cell_iterator cell_2 = cell_1++;
  Triangulation<2>::face_iterator face_1;
  Triangulation<2>::face_iterator face_2;

  // Look for the two outermost faces:
  for (const unsigned int j : GeometryInfo<2>::face_indices())
    {
      if (cell_1->face(j)->center()[1] > 2.9)
        face_1 = cell_1->face(j);
      if (cell_2->face(j)->center()[1] < -2.9)
        face_2 = cell_2->face(j);
    }
  face_1->set_boundary_id(42);
  face_2->set_boundary_id(43);

  triangulation.refine_global(1);
}


/* The 3D case */
void
generate_grid(Triangulation<3> &triangulation, int orientation)
{
  Point<3>              vertices_1[] = {Point<3>(-1., -1., -3.),
                                        Point<3>(+1., -1., -3.),
                                        Point<3>(-1., +1., -3.),
                                        Point<3>(+1., +1., -3.),
                                        Point<3>(-1., -1., -1.),
                                        Point<3>(+1., -1., -1.),
                                        Point<3>(-1., +1., -1.),
                                        Point<3>(+1., +1., -1.),
                                        Point<3>(-1., -1., +1.),
                                        Point<3>(+1., -1., +1.),
                                        Point<3>(-1., +1., +1.),
                                        Point<3>(+1., +1., +1.),
                                        Point<3>(-1., -1., +3.),
                                        Point<3>(+1., -1., +3.),
                                        Point<3>(-1., +1., +3.),
                                        Point<3>(+1., +1., +3.)};
  std::vector<Point<3>> vertices(&vertices_1[0], &vertices_1[16]);

  std::vector<CellData<3>> cells(2, CellData<3>());

  /* cell 0 */
  int cell_vertices_0[GeometryInfo<3>::vertices_per_cell] = {
    0, 1, 2, 3, 4, 5, 6, 7};

  /* cell 1 */
  int cell_vertices_1[8][GeometryInfo<3>::vertices_per_cell] = {
    {8, 9, 10, 11, 12, 13, 14, 15},
    {9, 11, 8, 10, 13, 15, 12, 14},
    {11, 10, 9, 8, 15, 14, 13, 12},
    {10, 8, 11, 9, 14, 12, 15, 13},
    {13, 12, 15, 14, 9, 8, 11, 10},
    {12, 14, 13, 15, 8, 10, 9, 11},
    {14, 15, 12, 13, 10, 11, 8, 9},
    {15, 13, 14, 12, 11, 9, 10, 8},
  };

  for (const unsigned int j : GeometryInfo<3>::vertex_indices())
    {
      cells[0].vertices[j] = cell_vertices_0[j];
      cells[1].vertices[j] = cell_vertices_1[orientation][j];
    }
  cells[0].material_id = 0;
  cells[1].material_id = 0;


  triangulation.create_triangulation(vertices, cells, SubCellData());

  Triangulation<3>::cell_iterator cell_1 = triangulation.begin();
  Triangulation<3>::cell_iterator cell_2 = cell_1++;
  Triangulation<3>::face_iterator face_1;
  Triangulation<3>::face_iterator face_2;

  // Look for the two outermost faces:
  for (const unsigned int j : GeometryInfo<3>::face_indices())
    {
      if (cell_1->face(j)->center()[2] > 2.9)
        face_1 = cell_1->face(j);
      if (cell_2->face(j)->center()[2] < -2.9)
        face_2 = cell_2->face(j);
    }
  face_1->set_boundary_id(42);
  face_2->set_boundary_id(43);

  triangulation.refine_global(1);
}



/*
 * Print out the face vertices as well as the orientation of a match:
 */
template <typename FaceIterator>
void
print_match(const FaceIterator                &face_1,
            const FaceIterator                &face_2,
            const types::geometric_orientation combined_orientation)
{
  static const int dim = FaceIterator::AccessorType::dimension;

  deallog << "face 1";
  for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_face; ++j)
    deallog << " :: " << face_1->vertex(j);
  deallog << std::endl;

  deallog << "face 2";
  for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_face; ++j)
    deallog << " :: " << face_2->vertex(j);
  deallog << std::endl;

  const auto [orientation, rotation, flip] =
    internal::split_face_orientation(combined_orientation);
  deallog << "orientation: " << orientation << "  flip: " << flip
          << "  rotation: " << rotation << std::endl
          << std::endl;
}

int
main()
{
  initlog();
  deallog << std::setprecision(4);
  deallog << "Test for 2D: Hypercube" << std::endl << std::endl;

  for (int i = 0; i < 2; ++i)
    {
      // Generate a triangulation and match:
      Triangulation<2> triangulation;

      generate_grid(triangulation, i);

      using CellIterator = Triangulation<2>::cell_iterator;
      using FaceVector = std::vector<GridTools::PeriodicFacePair<CellIterator>>;
      FaceVector test;
      GridTools::collect_periodic_faces(
        triangulation, 42, 43, 1, test, dealii::Tensor<1, 2>());

      deallog << "Triangulation: " << i << std::endl;

      for (FaceVector::iterator it = test.begin(); it != test.end(); ++it)
        print_match(it->cell[0]->face(it->face_idx[0]),
                    it->cell[1]->face(it->face_idx[1]),
                    it->orientation);
    }

  deallog << "Test for 3D: Hypercube" << std::endl << std::endl;

  for (int i = 0; i < 8; ++i)
    {
      // Generate a triangulation and match:
      Triangulation<3> triangulation;

      generate_grid(triangulation, i);

      using CellIterator = Triangulation<3>::cell_iterator;
      using FaceVector = std::vector<GridTools::PeriodicFacePair<CellIterator>>;
      FaceVector test;
      GridTools::collect_periodic_faces(
        triangulation, 42, 43, 2, test, dealii::Tensor<1, 3>());

      deallog << "Triangulation: " << i << std::endl;

      for (FaceVector::iterator it = test.begin(); it != test.end(); ++it)
        print_match(it->cell[0]->face(it->face_idx[0]),
                    it->cell[1]->face(it->face_idx[1]),
                    it->orientation);
    }

  return 0;
}
