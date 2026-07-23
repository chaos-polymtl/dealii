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

#include "../tests.h"

#include "fe_support_points_common.h"



int
main()
{
  initlog();

  CHECK_ALL(FaceNedelec, 0, 2);
  CHECK_ALL(FaceNedelec, 0, 3);
  CHECK_ALL(FaceNedelec, 1, 2);
  CHECK_ALL(FaceNedelec, 1, 3);
}
