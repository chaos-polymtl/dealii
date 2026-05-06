/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2026 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

// @sect3{Include files}

#include <deal.II/base/function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

namespace Step103
{
  using namespace dealii;


  template <int dim>
  class AnalyticalSolutionPressureReal : public Function<dim>
  {
  public:
    AnalyticalSolutionPressureReal(const double wavenumber, const double theta)
      : Function<dim>()
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    const double wavenumber;
    const double theta;
  };

  template <int dim>
  double AnalyticalSolutionPressureReal<dim>::value(
    const Point<dim> &p,
    const unsigned int /*component*/) const
  {
    return std::cos(wavenumber *
                    (p[0] * std::cos(theta) + p[1] * std::sin(theta)));
  }

  template <int dim>
  class DEM
  {
  public:
    DEM();
    void run(){};

  private:
    void setup_triangulation(){};
    void insert_particles(){};
    void output_results(unsigned int cycle){};


    Triangulation<dim> triangulation;


  };

int main()
{
  const unsigned int dim = 2;

  try
    {
      Step103::DPGHelmholtz<dim> dem();

      dem.run();
    }
  catch (std::exception &exc)
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
