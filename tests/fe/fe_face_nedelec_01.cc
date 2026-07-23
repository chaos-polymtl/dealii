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


// Basic sanity checks for FE_FaceNedelec: structural data (number of degrees of
// freedom per object, components, conformity, primitivity) and a round trip
// through FETools::get_fe_by_name().

#include <deal.II/fe/fe_face_nedelec.h>
#include <deal.II/fe/fe_tools.h>

#include "../tests.h"



template <int dim>
void
check(const unsigned int order)
{
  FE_FaceNedelec<dim> fe(order);

  deallog << fe.get_name() << std::endl;
  deallog << "  n_components         = " << fe.n_components() << std::endl;
  deallog << "  n_dofs_per_vertex    = " << fe.n_dofs_per_vertex() << std::endl;
  deallog << "  n_dofs_per_line      = " << fe.n_dofs_per_line() << std::endl;
  if (dim == 3)
    deallog << "  n_dofs_per_quad      = " << fe.n_dofs_per_quad(0)
            << std::endl;
  deallog << "  n_dofs_per_hex       = " << fe.n_dofs_per_hex() << std::endl;
  deallog << "  n_dofs_per_face      = " << fe.n_dofs_per_face(0) << std::endl;
  deallog << "  n_dofs_per_cell      = " << fe.n_dofs_per_cell() << std::endl;
  deallog << "  is_primitive         = " << (fe.is_primitive() ? "yes" : "no")
          << std::endl;
  deallog << "  conforms(Hcurl)      = "
          << (fe.conforms(FiniteElementData<dim>::Hcurl) ? "yes" : "no")
          << std::endl;

  // Round trip through get_fe_by_name.
  std::unique_ptr<FiniteElement<dim, dim>> fe2 =
    FETools::get_fe_by_name<dim, dim>(fe.get_name());
  deallog << "  get_fe_by_name       = " << fe2->get_name() << std::endl;
  Assert(fe2->get_name() == fe.get_name(), ExcInternalError());
}



int
main()
{
  initlog();

  for (unsigned int order = 0; order < 3; ++order)
    check<2>(order);
  for (unsigned int order = 0; order < 2; ++order)
    check<3>(order);

  deallog << "OK" << std::endl;
}
