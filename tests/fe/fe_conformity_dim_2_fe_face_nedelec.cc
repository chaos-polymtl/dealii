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

// Check the tangential continuity (H(curl) conformity) of FE_FaceNedelec in 2d
// across an interior face for all non-standard orientation configurations.

#include <deal.II/fe/fe_face_nedelec.h>

#include "../tests.h"

// STL
#include <fstream>
#include <iostream>

// My test headers
#include "fe_conformity_test.h"

#define PRECISION 4

int
main(int, char **)
{
  std::ofstream logfile("output");
  dealii::deallog << std::setprecision(PRECISION);
  dealii::deallog << std::fixed;
  logfile << std::setprecision(PRECISION);
  logfile << std::fixed;
  dealii::deallog.attach(logfile);

  try
    {
      using namespace FEConforimityTest;

      constexpr int dim = 2;

      // As for the reference test fe_conformity_dim_2_fe_nedelec, only the
      // lowest order is checked in 2d: FE_FaceNedelec shares the edge (line)
      // degrees of freedom and their sign handling with FE_Nedelec (see the
      // fe_face_nedelec_vs_nedelec test, which shows the shape functions are
      // identical), so the 2d conformity behaviour is exactly that of
      // FE_Nedelec, which this reference test exercises at degree 0. The
      // three-dimensional test additionally covers degree 1 (edge and face
      // degrees of freedom).
      for (unsigned int fe_degree = 0; fe_degree < 1; ++fe_degree)
        {
          // H(curl) conformal
          FE_FaceNedelec<dim> fe(fe_degree);

          {
            for (unsigned int this_switch = 0; this_switch < (dim == 2 ? 4 : 8);
                 ++this_switch)
              {
                deallog << std::endl
                        << "*******   degree " << fe_degree
                        << "   *******   orientation case " << this_switch
                        << "   *******" << std::endl;

                FEConformityTest<dim> fe_conformity_tester(fe, this_switch);
                fe_conformity_tester.run();
              }
          }
        } // ++fe_degree
    }
  catch (const std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
