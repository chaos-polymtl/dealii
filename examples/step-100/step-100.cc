/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2025 by the deal.II authors
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

// The DPG method requires a large breadth of element type which are included
// below:

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_trace.h>

// The rest of the includes are some well-known files:

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <fstream>
#include <iostream>

// @sect3{The <code>Step100</code> class}
namespace Step100
{
  using namespace dealii;


  // In the following function declaration, we will create the analytical
  // solutions of the velocity field ($u$) and the pressure field ($p$) and the
  // associated boundary values. However, in what follows, we will avoid to use
  // deal.ii complexe capabilities and only use the complex functions that are
  // defined in the C++ standard library. Consequently, we will define two
  // implementation of each functions, one for the real component and one for
  // the imaginary one.

  // Create analytical solution class for kinematic pressure (p).
  template <int dim>
  class AnalyticalSolution_p_real : public Function<dim>
  {
  public:
    // The analytical solution will depend on the wavenumber and the angle
    // defined for the direction of propagation so we will add them to the
    // constructor. The pressure is a scalar field, so we only need one
    // component by default.
    AnalyticalSolution_p_real(double wavenumber, double theta)
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
  double AnalyticalSolution_p_real<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    // Imaginary unit
    constexpr std::complex<double> imag(0., 1.);

    return std::exp(-imag * wavenumber *
                    (p[0] * std::cos(theta) + p[1] * std::sin(theta)))
      .real();
  }

  // The same goes for the imaginary part of the analytical solution
  template <int dim>
  class AnalyticalSolution_p_imag : public Function<dim>
  {
  public:
    AnalyticalSolution_p_imag(double wavenumber, double theta)
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
  double AnalyticalSolution_p_imag<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    constexpr std::complex<double> imag(0., 1.);

    return std::exp(-imag * wavenumber *
                    (p[0] * std::cos(theta) + p[1] * std::sin(theta)))
      .imag();
  }


  // Now we create analytical solution class for the velocity field (u)
  template <int dim>
  class AnalyticalSolution_u_real : public Function<dim>
  {
    // This class is similar to the previous ones but since the velocity field
    // is a vector, we will need dim component. For our problem of interest, dim
    // = 2.
  public:
    AnalyticalSolution_u_real(double wavenumber, double theta)
      : Function<dim>(2)
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
  double
  AnalyticalSolution_u_real<dim>::value(const Point<dim>  &p,
                                        const unsigned int component) const
  {
    constexpr std::complex<double> imag(0., 1.);

    if (component == 0)
      return (std::cos(theta) *
              std::exp(-imag * wavenumber *
                       (p[0] * std::cos(theta) + p[1] * std::sin(theta))))
        .real();
    else if (component == 1)
      return (std::sin(theta) *
              std::exp(-imag * wavenumber *
                       (p[0] * std::cos(theta) + p[1] * std::sin(theta))))
        .real();
    else
      throw std::runtime_error(
        "Too much components for the analytical solution");
  }

  // The same goes for the imaginary part of the analytical solution of the
  // velocity field
  template <int dim>
  class AnalyticalSolution_u_imag : public Function<dim>
  {
  public:
    AnalyticalSolution_u_imag(double wavenumber, double theta)
      : Function<dim>(2)
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
  double
  AnalyticalSolution_u_imag<dim>::value(const Point<dim>  &p,
                                        const unsigned int component) const
  {
    constexpr std::complex<double> imag(0., 1.);

    if (component == 0)
      return (std::cos(theta) *
              std::exp(-imag * wavenumber *
                       (p[0] * std::cos(theta) + p[1] * std::sin(theta))))
        .imag();
    else if (component == 1)
      return (std::sin(theta) *
              std::exp(-imag * wavenumber *
                       (p[0] * std::cos(theta) + p[1] * std::sin(theta))))
        .imag();
    else
      throw std::runtime_error(
        "Too much components for the analytical solution");
  }

  // Now we will do a similar job for the boundary values functions. The main
  // difference is that the number of components will now be 4 because those
  // functions will be applied to our space of skeletons unknowns which are
  // scalars for the pressure and velocity fields, but also have real and
  // imaginary parts.
  template <int dim>
  class BoundaryValues_p_real : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_p_real(const double wavenumber, const double theta)
      : Function<dim>(4)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    double wavenumber;
    double theta;
  };

  template <int dim>
  double BoundaryValues_p_real<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    constexpr std::complex<double> imag(0., 1.);

    return std::exp(-imag * wavenumber * p[1] * std::sin(theta)).real();
  }

  template <int dim>
  class BoundaryValues_p_imag : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_p_imag(const double wavenumber, const double theta)
      : Function<dim>(4)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    double wavenumber;
    double theta;
  };

  template <int dim>
  double BoundaryValues_p_imag<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    constexpr std::complex<double> imag(0., 1.);

    return std::exp(-imag * wavenumber * p[1] * std::sin(theta)).imag();
  }

  // Lastly, we create the boundary values for the velocity field $\hat{u}_n =
  // \mathb{u} \cdot n$.
  template <int dim>
  class BoundaryValues_u_real : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_u_real(const double wavenumber, const double theta)
      : Function<dim>(4)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    double wavenumber;
    double theta;
  };

  template <int dim>
  double BoundaryValues_u_real<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    constexpr std::complex<double> imag(0., 1.);
    return -1 * (std::sin(theta) *
                 std::exp(-imag * wavenumber * p[0] * std::cos(theta)))
                  .real();
  }
  template <int dim>
  class BoundaryValues_u_imag : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_u_imag(const double wavenumber, const double theta)
      : Function<dim>(4)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    double wavenumber;
    double theta;
  };

  template <int dim>
  double BoundaryValues_u_imag<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    constexpr std::complex<double> imag(0., 1.);

    return -1 * (std::sin(theta) *
                 std::exp(-imag * wavenumber * p[0] * std::cos(theta)))
                  .imag();
  }

  // @sect3{The <code>DPGHelmholtz</code> class template}

  // Next let's declare the main class of this program. The structure follows
  // that of usual programs. The main difference lies in the fact that we rely
  // on multiple DOFHandler and FESystem. The DOFHandlers that we rely on are
  // the following:
  // - The <code>dof_handler_trial_interior</code> is for the unknowns in the
  // interior of the cells
  // - The <code>dof_handler_trial_skeleton</code> is for the unknowns in the
  // skeleton
  // - The <code>dof_handler_test</code> is for the test functions. Although we
  // do not use the unknowns associated with this DOFHandler, it enables us to
  // evaluate the test function we will use in DPG.
  // The same applies for the three FESystem:
  // <code>fe_system_trial_interior</code>,
  // <code>fe_system_trial_skeleton</code> and <code>fe_system_test</code>. In
  // each one of these we will store the relevant finite element space in the
  // same order to avoid comfusion. The first component will therefore always be
  // related to the real part of the velocity, the second component to the its
  // imaginary part, the third component to the real part of the pressure and
  // the fourth component to its imaginary part.

  template <int dim>
  class DPGHelmholtz
  {
  public:
    // The constructor takes as an argument the degree of the trial space as
    // well as the delta degree between the trial space and the test space which
    // is necesasry to constructor the DPG problem. The
    // <code>delta_degree</code> must be at least 1 to ensure that the DPG
    // method is functional. The parameter <code>theta</code> determines the
    // angle of the incident plane wave. The angle must between $0$ and
    // $\frac{\pi}{2}$ included. Those restrictions are asserted in the
    // constructor.

    DPGHelmholtz(unsigned int degree,
                 unsigned int delta_degree,
                 double       wavenumber,
                 double       theta);
    void run();

  private:
    // The setup_system function initializes the three DoFHandlers, the system
    // matrix and right-hand side and establishes the boundary conditions that
    // rely on constraints.

    void setup_system();

    // The assemble_system assembles both the right-hand side and the system
    // matrix. This function is used twice per resolution and it has two
    // functions.
    // - When <code>solve_interior = false</code> the system is assembled and is
    // locally condensed such that the resulting system only contains the
    // skeleton uknowns. This is achieved by local condensation.
    // - When <code>solve_interior = true</code> the system is assembled and the
    // skeleton degrees of freedom are used to reconstruct the interior
    // solution.

    void assemble_system(bool solve_interior = false);

    // Solves the linear system of equation. This linear system of equation is
    // only for the skeleton unknowns.
    void solve_skeleton();

    // Refines the mesh uniformly
    void refine_grid(unsigned int cycle);

    // Write the skeleton and the interior unknowns into two different paraview
    // files.
    void output_results(unsigned int cycle);

    // Calculates the $L^2$ norm of the error using the analytical solution.
    void calculate_L2_error();

    Triangulation<dim> triangulation;

    // Variables for the interior
    const FESystem<dim> fe_trial_interior;
    DoFHandler<dim>     dof_handler_trial_interior;
    Vector<double>      solution_interior;

    // Variables for the skeleton and, consequently, the system
    const FESystem<dim>       fe_trial_skeleton;
    DoFHandler<dim>           dof_handler_trial_skeleton;
    Vector<double>            solution_skeleton;
    Vector<double>            system_rhs;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;
    AffineConstraints<double> constraints;

    // Variables for the test space
    const FESystem<dim> fe_test;
    DoFHandler<dim>     dof_handler_test;

    // Container for the L2 error and other related quantities
    ConvergenceTable error_table;

    // Coefficient which are used to define the problem
    const double wavenumber;
    const double theta;

    // Define exctractors that will be used at multiple places to select the
    // relevant components for the calculation. Those can be created at the
    // class level since they only depend on the FEM problem we want to solve
    // and how those specific components are defined in the finite element
    // space. We therefore follow the same nomanclature as described above.
    const FEValuesExtractors::Vector extractor_u_real;
    const FEValuesExtractors::Vector extractor_u_imag;
    const FEValuesExtractors::Scalar extractor_p_real;
    const FEValuesExtractors::Scalar extractor_p_imag;

    // However, the skeleton space does not have the same number of components
    // because the space $H^{-1/2}$ related to the velocity field is a scalar
    // field. Consequently, we define the following extractors for the skeleton
    // space:
    const FEValuesExtractors::Scalar extractor_u_hat_real;
    const FEValuesExtractors::Scalar extractor_u_hat_imag;
    const FEValuesExtractors::Scalar extractor_p_hat_real;
    const FEValuesExtractors::Scalar extractor_p_hat_imag;
  };

  // @sect3{DPGHelmholtz Constructor}
  // The Q elements have a degree higher than the others because their
  // numerotation start at 1 instead of 0.

  template <int dim>
  DPGHelmholtz<dim>::DPGHelmholtz(const unsigned int degree,
                                  const unsigned int delta_degree,
                                  double             wavenumber,
                                  double             theta)
    : fe_trial_interior(FE_DGQ<dim>(degree) ^ dim,
                        FE_DGQ<dim>(degree) ^ dim,
                        FE_DGQ<dim>(degree),
                        FE_DGQ<dim>(degree))
    , // (u, u_imag, p, p_imag)
    dof_handler_trial_interior(triangulation)
    , fe_trial_skeleton(FE_FaceQ<dim>(degree),
                        FE_FaceQ<dim>(degree),
                        FE_TraceQ<dim>(degree + 1),
                        FE_TraceQ<dim>(degree + 1))
    , // (u_hat_n, u_hat_n_imag, p_hat, p_hat_imag)
    dof_handler_trial_skeleton(triangulation)
    , fe_test(FE_RaviartThomas<dim>(degree + delta_degree),
              FE_RaviartThomas<dim>(degree + delta_degree),
              FE_Q<dim>(degree + delta_degree + 1),
              FE_Q<dim>(degree + delta_degree + 1))
    , // (v, v_imag, q, q_imag)
    dof_handler_test(triangulation)
    , wavenumber(wavenumber)
    , theta(theta)

    // Here we initialize the FEValuesExtractors that will be used
    , extractor_u_real(0)
    , extractor_u_imag(dim)
    , extractor_p_real(2 * dim)
    , extractor_p_imag(2 * dim + 1)

    , extractor_u_hat_real(0)
    , extractor_u_hat_imag(1)
    , extractor_p_hat_real(2)
    , extractor_p_hat_imag(3)
  {
    // Here we check is everything is correctly defined for our problem to work.
    // The step is only implemented for the 2D case, so we verify the dimension.
    AssertDimension(dim, 2);

    // The degree of the test space must be at least one degree higher than the
    // trial space so the delta_degree variable needs to be at least 1.
    Assert(delta_degree >= 1,
           ExcMessage("The delta_degree needs to be at least 1."));

    // The wavenumber is the magnitude of the wave vector and must be positive.
    Assert(wavenumber > 0, ExcMessage("The wavenumber must be positive."));

    // The angle theta must be in the interval [0, pi/2]. Other angles are
    // redundant and would not be compatible with the current boundary
    // definitions.
    Assert(theta >= 0 && theta <= M_PI / 2,
           ExcMessage("The angle theta must be in the interval [0, pi/2]."));
  }

  // @sect3{DPG::setup_system}
  // This function is similar to the other examples. The main difference lies in
  // the fact that we need to setup multiple DOFHandlers for the interior, the
  // skeleton and the test space.
  template <int dim>
  void DPGHelmholtz<dim>::setup_system()
  {
    dof_handler_trial_skeleton.distribute_dofs(fe_trial_skeleton);
    dof_handler_trial_interior.distribute_dofs(fe_trial_interior);
    dof_handler_test.distribute_dofs(fe_test);

    // We print the number of degree of freedoms for each of the DoFHandler as
    // well as adding this information to the ConvergenceTable.

    std::cout << std::endl
              << "Number of dofs for the interior: "
              << dof_handler_trial_interior.n_dofs() << std::endl;

    error_table.add_value("dofs_interior", dof_handler_trial_interior.n_dofs());

    std::cout << "Number of dofs for the skeleton: "
              << dof_handler_trial_skeleton.n_dofs() << std::endl;

    error_table.add_value("dofs_skeleton", dof_handler_trial_skeleton.n_dofs());

    std::cout << "Number of dofs for the test space: "
              << dof_handler_test.n_dofs() << std::endl;

    error_table.add_value("dofs_test", dof_handler_test.n_dofs());


    constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler_trial_skeleton,
                                            constraints);


    // We need to specify different boundary conditions for the four unknowns on
    // the faces. Therefore, we instantiate the functions that are used to
    // establish these boundary conditions.
    BoundaryValues_p_real<dim> p_real(wavenumber, theta);
    BoundaryValues_p_imag<dim> p_imag(wavenumber, theta);
    BoundaryValues_u_real<dim> u_real(wavenumber, theta);
    BoundaryValues_u_imag<dim> u_imag(wavenumber, theta);

    // Using the functions and the FEValuesExtractors, we impose the four
    // different constraints. As stated in the problem description, we first
    // impose a Dirichlet boundary condition on the pressure field for the left
    // boundary (id=0).
    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             0,
                                             p_real,
                                             constraints,
                                             fe_trial_skeleton.component_mask(
                                               extractor_p_hat_real));
    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             0,
                                             p_imag,
                                             constraints,
                                             fe_trial_skeleton.component_mask(
                                               extractor_p_hat_imag));

    // Then we impose a Neumann boundary condition on pressure by applying a
    // Dirichlet on the pressure "flux", which is the normal velocity field on
    // the bottom boundary (id=2).
    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             2,
                                             u_real,
                                             constraints,
                                             fe_trial_skeleton.component_mask(
                                               extractor_u_hat_real));
    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             2,
                                             u_imag,
                                             constraints,
                                             fe_trial_skeleton.component_mask(
                                               extractor_u_hat_imag));
    constraints.close();

    // The linear system that we form is only related to the skeleton unknowns.
    // We initialize all the necessary variables.
    solution_skeleton.reinit(dof_handler_trial_skeleton.n_dofs());
    system_rhs.reinit(dof_handler_trial_skeleton.n_dofs());
    solution_interior.reinit(dof_handler_trial_interior.n_dofs());

    DynamicSparsityPattern dsp(dof_handler_trial_skeleton.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_trial_skeleton,
                                    dsp,
                                    constraints,
                                    false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  };

  // @sect3{DPG::assemble_system}

  template <int dim>
  void DPGHelmholtz<dim>::assemble_system(const bool solve_interior)
  {
    // Define quadrature rules and related variables
    const QGauss<dim> quadrature_formula(
      fe_test.degree +
      1); // The quadrature formula should be the same for both trial and test
          // and the test have higher polynomial degree so we use it
    const QGauss<dim - 1> face_quadrature_formula(fe_test.degree + 1);
    const unsigned int    n_q_points      = quadrature_formula.size();
    const unsigned int    n_face_q_points = face_quadrature_formula.size();

    // Update the FEValues
    FEValues<dim> fe_values_trial_interior(fe_trial_interior,
                                           quadrature_formula,
                                           update_values |
                                             update_quadrature_points |
                                             update_JxW_values);

    FEValues<dim> fe_values_test(fe_test,
                                 quadrature_formula,
                                 update_values | update_gradients |
                                   update_quadrature_points);

    FEFaceValues<dim> fe_values_trial_skeleton(fe_trial_skeleton,
                                               face_quadrature_formula,
                                               update_values |
                                                 update_quadrature_points |
                                                 update_normal_vectors |
                                                 update_JxW_values);

    FEFaceValues<dim> fe_face_values_test(fe_test,
                                          face_quadrature_formula,
                                          update_values |
                                            update_quadrature_points);


    // Get number of dofs for each type of element
    const unsigned int dofs_per_cell_test = fe_test.n_dofs_per_cell();
    const unsigned int dofs_per_cell_trial_interior =
      fe_trial_interior.n_dofs_per_cell();
    const unsigned int dofs_per_cell_trial_skeleton =
      fe_trial_skeleton.n_dofs_per_cell();

    // First we create the system before condensation
    // We create the DPG local matrices
    LAPACKFullMatrix<double> G_matrix(dofs_per_cell_test, dofs_per_cell_test);
    LAPACKFullMatrix<double> B_matrix(dofs_per_cell_test,
                                      dofs_per_cell_trial_interior);
    LAPACKFullMatrix<double> B_hat_matrix(dofs_per_cell_test,
                                          dofs_per_cell_trial_skeleton);
    LAPACKFullMatrix<double> D_matrix(dofs_per_cell_trial_skeleton,
                                      dofs_per_cell_trial_skeleton);

    // We create the DPG local vectors
    Vector<double> g_vector(dofs_per_cell_trial_skeleton);
    Vector<double> l_vector(dofs_per_cell_test);

    // When building the different matrices, we will need the shapes functions
    // values, gradient and divergence at the quadrature points for the trial
    // and test spaces. To avoid the query of the FEValues at each quadrature
    // point, we will use containers that will store the desired values before
    // hand.

    // We create the condensation matrices
    LAPACKFullMatrix<double> M1_matrix(dofs_per_cell_trial_interior,
                                       dofs_per_cell_trial_interior);
    LAPACKFullMatrix<double> M2_matrix(dofs_per_cell_trial_interior,
                                       dofs_per_cell_trial_skeleton);
    LAPACKFullMatrix<double> M3_matrix(dofs_per_cell_trial_skeleton,
                                       dofs_per_cell_trial_skeleton);
    LAPACKFullMatrix<double> M4_matrix(dofs_per_cell_trial_interior,
                                       dofs_per_cell_test);
    LAPACKFullMatrix<double> M5_matrix(dofs_per_cell_trial_skeleton,
                                       dofs_per_cell_test);

    // During the calculation, we require intermediary matrices that we allocate
    // here.
    LAPACKFullMatrix<double> tmp_matrix(dofs_per_cell_trial_skeleton,
                                        dofs_per_cell_trial_interior);

    LAPACKFullMatrix<double> tmp_matrix2(dofs_per_cell_trial_skeleton,
                                         dofs_per_cell_trial_skeleton);

    LAPACKFullMatrix<double> tmp_matrix3(dofs_per_cell_trial_skeleton,
                                         dofs_per_cell_test);

    // We also require a temporary condensation vector.
    Vector<double> tmp_vector(dofs_per_cell_trial_interior);

    // We create the matrix and the rhs that will be distributed in the full
    // system
    FullMatrix<double> cell_matrix(dofs_per_cell_trial_skeleton,
                                   dofs_per_cell_trial_skeleton);
    Vector<double>     cell_skeleton_rhs(dofs_per_cell_trial_skeleton);

    // Finally, when reconstructing the interior solution from the skeleton, we
    // require additional vectors that we allocate here.
    Vector<double> cell_interior_rhs(dofs_per_cell_trial_interior);
    Vector<double> cell_interior_solution(dofs_per_cell_trial_interior);
    Vector<double> cell_skeleton_solution(dofs_per_cell_trial_skeleton);

    // Create the dofs indices mapping container
    // We recall that the final unknowns of the system are the skeleton
    // unknowns.
    std::vector<types::global_dof_index> local_dof_indices(
      dofs_per_cell_trial_skeleton);

    // We also define the imaginary unit and two complex constant that will be
    // used during the following assembly. Note that even if the system and
    // matrix that we build are real, we still make use of the standard library
    // complex operation to facilitate some computations as it is done in
    // step-81.
    constexpr std::complex<double> imag(0., 1.);
    const std::complex<double>     iomega      = imag * wavenumber;
    const std::complex<double>     conj_iomega = conj(iomega);

    // We first loop over the cells of the triangulation. We will choose to loop
    // on using the DofHandler of the trial space because it is the natural
    // choice since it is where we will compute our the solution.
    for (const auto &cell : dof_handler_trial_interior.active_cell_iterators())
      {
        // We first reinitialize the FEValues objects to the current cell.
        fe_values_trial_interior.reinit(cell);

        // However, we will also need to reinitialize the FEValues for the test
        // space and make sure that is the same cell as used for the trial
        // space.
        const typename DoFHandler<dim>::active_cell_iterator cell_test =
          cell->as_dof_handler_iterator(dof_handler_test);
        fe_values_test.reinit(cell_test);

        // Similarly, we reinitialize the FEValues for the trial space on the
        // skeleton, but this will not be used before we also loop on the cells
        // faces.
        const typename DoFHandler<dim>::active_cell_iterator cell_skeleton =
          cell->as_dof_handler_iterator(dof_handler_trial_skeleton);

        // We then reinitialize all the matrices that we are aggregating
        // information for each cell.
        G_matrix     = 0;
        B_matrix     = 0;
        B_hat_matrix = 0;
        D_matrix     = 0;
        g_vector     = 0;
        l_vector     = 0;

        // We also need to do it for the condensation matrices ?
        M1_matrix = 0;
        M2_matrix = 0;
        M3_matrix = 0;
        M4_matrix = 0;
        M5_matrix = 0;

        // Loop over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double &JxW = fe_values_trial_interior.JxW(q_point);

            // Loop over test space dofs
            for (const auto i : fe_values_test.dof_indices())
              {
                // Define the necessary complex test basis functions
                const auto v_i_conj =
                  fe_values_test[extractor_u_real].value(i, q_point) -
                  imag * fe_values_test[extractor_u_imag].value(i, q_point);

                const auto v_i_div_conj =
                  fe_values_test[extractor_u_real].divergence(i, q_point) -
                  imag *
                    fe_values_test[extractor_u_imag].divergence(i, q_point);

                const auto q_i_conj =
                  fe_values_test[extractor_p_real].value(i, q_point) -
                  imag * fe_values_test[extractor_p_imag].value(i, q_point);

                const auto q_i_grad_conj =
                  fe_values_test[extractor_p_real].gradient(i, q_point) -
                  imag * fe_values_test[extractor_p_imag].gradient(i, q_point);

                // Get the information on witch element the dof is
                const unsigned int current_element_test_i =
                  fe_test.system_to_base_index(i).first.first;

                // If in Q element -> test for pressure
                if ((current_element_test_i == 2) ||
                    (current_element_test_i == 3))
                  {
                    // Compute load vector
                    l_vector(i) += 0;
                  }

                // Construct G_matrix, loop over test space dofs again
                for (const auto j : fe_values_test.dof_indices())
                  {
                    // Create the test basis functions
                    const auto v_j =
                      fe_values_test[extractor_u_real].value(j, q_point) +
                      imag * fe_values_test[extractor_u_imag].value(j, q_point);

                    const auto v_j_div =
                      fe_values_test[extractor_u_real].divergence(j, q_point) +
                      imag *
                        fe_values_test[extractor_u_imag].divergence(j, q_point);

                    const auto q_j =
                      fe_values_test[extractor_p_real].value(j, q_point) +
                      imag * fe_values_test[extractor_p_imag].value(j, q_point);

                    const auto q_j_grad =
                      fe_values_test[extractor_p_real].gradient(j, q_point) +
                      imag *
                        fe_values_test[extractor_p_imag].gradient(j, q_point);

                    // Get the information on witch element the dof is
                    const unsigned int current_element_test_j =
                      fe_test.system_to_base_index(j).first.first;

                    // If both Raviart-thomas element
                    if (((current_element_test_i == 0) ||
                         (current_element_test_i == 1)) &&
                        ((current_element_test_j == 0) ||
                         (current_element_test_j == 1)))
                      {
                        // (v,v*) + (div(v),div(v)*) + (i omega v, (i
                        // omega v)*)
                        G_matrix(i, j) +=
                          (((v_j * v_i_conj) + (v_j_div * v_i_div_conj) +
                            (conj_iomega * v_j * iomega * v_i_conj)) *
                           JxW)
                            .real();
                      }
                    // If in i Raviart-thomas element and j in Q element
                    else if (((current_element_test_i == 0) ||
                              (current_element_test_i == 1)) &&
                             ((current_element_test_j == 2) ||
                              (current_element_test_j == 3)))
                      {
                        // (grad(q), (i omega v)*) + (i omega q, div(v) *)
                        G_matrix(i, j) -=
                          (((q_j_grad * iomega * v_i_conj) +
                            (conj_iomega * q_j * v_i_div_conj)) *
                           JxW)
                            .real();
                      }
                    // If in i Q element and j in Raviart-thomas element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_test_j == 0) ||
                              (current_element_test_j == 1)))
                      {
                        // ( i omega v , grad(q)*) + (div(v), (i omega v) *)
                        G_matrix(i, j) -=
                          (((conj_iomega * v_j * q_i_grad_conj) +
                            (v_j_div * iomega * q_i_conj)) *
                           JxW)
                            .real();
                      }
                    // If both Q element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_test_j == 2) ||
                              (current_element_test_j == 3)))
                      {
                        // (q,q*) + (grad(q),grad(q)*) + (i omega q, (i omega
                        // v)*)
                        G_matrix(i, j) +=
                          (((q_j * q_i_conj) + (q_j_grad * q_i_grad_conj) +
                            (conj_iomega * q_j * iomega * q_i_conj)) *
                           JxW)
                            .real();
                      }
                  }

                // Loop over trial space dofs
                for (const auto j : fe_values_trial_interior.dof_indices())
                  {
                    // Create the trial basis functions
                    const auto u_j =
                      fe_values_trial_interior[extractor_u_real].value(
                        j, q_point) +
                      imag * fe_values_trial_interior[extractor_u_imag].value(
                               j, q_point);

                    const auto p_j =
                      fe_values_trial_interior[extractor_p_real].value(
                        j, q_point) +
                      imag * fe_values_trial_interior[extractor_p_imag].value(
                               j, q_point);

                    // Get the information to map the index to the right shape
                    // function
                    const unsigned int current_element_trial_j =
                      fe_trial_interior.system_to_base_index(j).first.first;

                    // If in Raviart-thomas element and DGQ^dim element
                    if (((current_element_test_i == 0) ||
                         (current_element_test_i == 1)) &&
                        ((current_element_trial_j == 0) ||
                         (current_element_trial_j == 1)))
                      {
                        // (i omega u, v*)
                        B_matrix(i, j) +=
                          ((iomega * u_j * v_i_conj) * JxW).real();
                      }
                    // If in Raviart-thomas element and DGQ element
                    else if (((current_element_test_i == 0) ||
                              (current_element_test_i == 1)) &&
                             ((current_element_trial_j == 2) ||
                              (current_element_trial_j == 3)))
                      {
                        // -(p,div(v)*)
                        B_matrix(i, j) -= ((p_j * v_i_div_conj) * JxW).real();
                      }

                    // If in Q element and DGQ^dim element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_trial_j == 0) ||
                              (current_element_trial_j == 1)))
                      {
                        // -(u,grad(q)*)
                        B_matrix(i, j) -= ((u_j * q_i_grad_conj) * JxW).real();
                      }
                    // If in Q element and DGQ element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_trial_j == 2) ||
                              (current_element_trial_j == 3)))
                      {
                        // (i omega p, q*)
                        B_matrix(i, j) +=
                          ((iomega * p_j * q_i_conj) * JxW).real();
                      }
                  }
              }
          }
        // Loop over all face
        for (const auto &face : cell->face_iterators())
          {
            // Reinitialization
            fe_face_values_test.reinit(cell, face);
            fe_values_trial_skeleton.reinit(cell_skeleton, face);

            // Get face number
            const auto face_no = cell->face_iterator_to_index(face);

            // Loop over all face quadrature points
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
              {
                // Get normal vector
                const auto &normal =
                  fe_values_trial_skeleton.normal_vector(q_point);
                const double &JxW_face = fe_values_trial_skeleton.JxW(q_point);

                // Loop over the test space dofs
                for (const auto i : fe_face_values_test.dof_indices())
                  {
                    // Create the face test basis functions
                    const auto v_n_i_conj =
                      normal *
                      (fe_face_values_test[extractor_u_real].value(i, q_point) -
                       imag *
                         fe_face_values_test[extractor_u_imag].value(i,
                                                                     q_point));

                    const auto q_i_conj =
                      fe_face_values_test[extractor_p_real].value(i, q_point) -
                      imag *
                        fe_face_values_test[extractor_p_imag].value(i, q_point);

                    // Get the information to map the index to the right shape
                    // function
                    const unsigned int current_element_test_i =
                      fe_test.system_to_base_index(i).first.first;

                    // Loop over trial space dofs
                    for (const auto j : fe_values_trial_skeleton.dof_indices())
                      {
                        // Create the face trial basis functions
                        const auto u_hat_n_j =
                          fe_values_trial_skeleton[extractor_u_hat_real].value(
                            j, q_point) +
                          imag * fe_values_trial_skeleton[extractor_u_hat_imag]
                                   .value(j, q_point);

                        const auto p_hat_j =
                          fe_values_trial_skeleton[extractor_p_hat_real].value(
                            j, q_point) +
                          imag * fe_values_trial_skeleton[extractor_p_hat_imag]
                                   .value(j, q_point);

                        // Get the information to map the index to the right
                        // shape function
                        const unsigned int current_element_trial_j =
                          fe_trial_skeleton.system_to_base_index(j).first.first;

                        // If in Raviart-thomas element and FE_FaceQ for p
                        if (((current_element_test_i == 0) ||
                             (current_element_test_i == 1)) &&
                            ((current_element_trial_j == 2) ||
                             (current_element_trial_j == 3)))
                          {
                            B_hat_matrix(i, j) +=
                              ((p_hat_j * v_n_i_conj) * JxW_face).real();
                          }

                        // If in Q element and FE_FaceQ for u_n
                        else if (((current_element_test_i == 2) ||
                                  (current_element_test_i == 3)) &&
                                 ((current_element_trial_j == 0) ||
                                  (current_element_trial_j == 1)))
                          {
                            // Get the neighbor cell id
                            int neighbor_cell_id = -1;
                            if (face->at_boundary())
                              {
                                neighbor_cell_id = INT_MAX;
                              }
                            else
                              {
                                neighbor_cell_id =
                                  cell->neighbor(face_no)->index();
                              }

                            // Get current cell id
                            const auto current_cell_id = cell->index();

                            // Initialize the flux orientation
                            double flux_orientation = 0.;
                            if (neighbor_cell_id > current_cell_id)
                              {
                                flux_orientation = 1.;
                              }
                            else
                              {
                                flux_orientation = -1.;
                              }

                            // (u_hat_n, q*)
                            B_hat_matrix(i, j) +=
                              (flux_orientation * u_hat_n_j * q_i_conj *
                               JxW_face)
                                .real();
                          }
                      }
                  }
              }

            // Build the robin boundary conditions
            if (face->at_boundary() &&
                ((face->boundary_id() == 1) || (face->boundary_id() == 3)))
              {
                // Boundary wavenumber ratio
                double k_ratio;
                if (face->boundary_id() == 1)
                  {
                    k_ratio = cos(theta);
                  }
                else if (face->boundary_id() == 3)
                  {
                    k_ratio = sin(theta);
                  }

                // Loop over all face quadrature points
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     ++q_point)
                  {
                    // Initialize reusable variables
                    const auto &normal =
                      fe_values_trial_skeleton.normal_vector(q_point);
                    const double JxW_face =
                      fe_values_trial_skeleton.JxW(q_point);
                    const double flux_orientation = 1.;

                    // Update the G_matrix
                    for (const auto i : fe_face_values_test.dof_indices())
                      {
                        // Create the face test basis functions
                        const auto v_n_i_conj =
                          normal * (fe_face_values_test[extractor_u_real].value(
                                      i, q_point) -
                                    imag * fe_face_values_test[extractor_u_imag]
                                             .value(i, q_point));

                        const auto q_i_conj =
                          fe_face_values_test[extractor_p_real].value(i,
                                                                      q_point) -
                          imag * fe_face_values_test[extractor_p_imag].value(
                                   i, q_point);

                        const unsigned int current_element_test_i =
                          fe_test.system_to_base_index(i).first.first;

                        for (const auto j : fe_face_values_test.dof_indices())
                          {
                            // Create the face test basis functions
                            const auto v_n_j =
                              normal *
                              (fe_face_values_test[extractor_u_real].value(
                                 j, q_point) +
                               imag * fe_face_values_test[extractor_u_imag]
                                        .value(j, q_point));

                            const auto q_j =
                              fe_face_values_test[extractor_p_real].value(
                                j, q_point) +
                              imag * fe_face_values_test[extractor_p_imag]
                                       .value(j, q_point);

                            const unsigned int current_element_test_j =
                              fe_test.system_to_base_index(j).first.first;

                            if (((current_element_test_i == 0) ||
                                 (current_element_test_i == 1)) &&
                                ((current_element_test_j == 0) ||
                                 (current_element_test_j == 1)))
                              {
                                // (v_n_j, v_n_i*)
                                G_matrix(i, j) +=
                                  (v_n_j * v_n_i_conj * JxW_face).real();
                              }
                            else if (((current_element_test_i == 0) ||
                                      (current_element_test_i == 1)) &&
                                     ((current_element_test_j == 2) ||
                                      (current_element_test_j == 3)))
                              {
                                // (k_n/ k * q_j, v_n_i*)
                                G_matrix(i, j) +=
                                  (k_ratio * q_j * v_n_i_conj * JxW_face)
                                    .real();
                              }
                            else if (((current_element_test_i == 2) ||
                                      (current_element_test_i == 3)) &&
                                     ((current_element_test_j == 0) ||
                                      (current_element_test_j == 1)))
                              {
                                // (v_n_j, k_n/ k * q_i_conj)
                                G_matrix(i, j) +=
                                  (v_n_j * k_ratio * q_i_conj * JxW_face)
                                    .real();
                              }
                            else if (((current_element_test_i == 2) ||
                                      (current_element_test_i == 3)) &&
                                     ((current_element_test_j == 2) ||
                                      (current_element_test_j == 3)))
                              {
                                // (k_n/ k * q_j, k_n/ k * q_i_conj)
                                G_matrix(i, j) += (k_ratio * q_j * k_ratio *
                                                   q_i_conj * JxW_face)
                                                    .real();
                              }
                          }
                      }

                    // Update the g_vector and D_matrix
                    for (const auto i : fe_values_trial_skeleton.dof_indices())
                      {
                        // Create the face trial basis functions
                        const auto u_hat_n_i_conj =
                          fe_values_trial_skeleton[extractor_u_hat_real].value(
                            i, q_point) -
                          imag * fe_values_trial_skeleton[extractor_u_hat_imag]
                                   .value(i, q_point);

                        const auto p_hat_i_conj =
                          fe_values_trial_skeleton[extractor_p_hat_real].value(
                            i, q_point) -
                          imag * fe_values_trial_skeleton[extractor_p_hat_imag]
                                   .value(i, q_point);

                        // Get the information to map the index to the right
                        // shape function
                        const unsigned int current_element_trial_i =
                          fe_trial_skeleton.system_to_base_index(i).first.first;

                        // No source terms, Sommerfeld B.C., so g_vector is zero
                        if ((current_element_trial_i == 0) ||
                            (current_element_trial_i == 1))
                          {
                            g_vector(i) -=
                              ((0) * u_hat_n_i_conj).real() * JxW_face;
                          }
                        else if ((current_element_trial_i == 2) ||
                                 (current_element_trial_i == 3))
                          {
                            g_vector(i) +=
                              ((0.) * k_ratio * p_hat_i_conj).real() * JxW_face;
                          }

                        // Loop over trial space dofs
                        for (const auto j :
                             fe_values_trial_skeleton.dof_indices())
                          {
                            // Create the face trial basis functions
                            const auto u_hat_n_j =
                              fe_values_trial_skeleton[extractor_u_hat_real]
                                .value(j, q_point) +
                              imag *
                                fe_values_trial_skeleton[extractor_u_hat_imag]
                                  .value(j, q_point);

                            const auto p_hat_j =
                              fe_values_trial_skeleton[extractor_p_hat_real]
                                .value(j, q_point) +
                              imag *
                                fe_values_trial_skeleton[extractor_p_hat_imag]
                                  .value(j, q_point);

                            // Get the information to map the index to the right
                            // shape function
                            const unsigned int current_element_trial_j =
                              fe_trial_skeleton.system_to_base_index(j)
                                .first.first;

                            // If in FE element and FE_FaceQ for p
                            if (((current_element_trial_i == 0) ||
                                 (current_element_trial_i == 1)) &&
                                ((current_element_trial_j == 0) ||
                                 (current_element_trial_j == 1)))
                              {
                                // -(u_hat_n_j, u_hat_n_i*)
                                D_matrix(i, j) -=
                                  (flux_orientation * u_hat_n_j *
                                   flux_orientation * u_hat_n_i_conj * JxW_face)
                                    .real();
                              }
                            else if (((current_element_trial_i == 0) ||
                                      (current_element_trial_i == 1)) &&
                                     ((current_element_trial_j == 2) ||
                                      (current_element_trial_j == 3)))
                              {
                                // (k_n/ k * p_hat_j , u_hat_n_i*)
                                D_matrix(i, j) +=
                                  (k_ratio * p_hat_j * flux_orientation *
                                   u_hat_n_i_conj * JxW_face)
                                    .real();
                              }
                            else if (((current_element_trial_i == 2) ||
                                      (current_element_trial_i == 3)) &&
                                     ((current_element_trial_j == 0) ||
                                      (current_element_trial_j == 1)))
                              {
                                // (u_hat_n_j, k_n/ k * p_hat_i_conj )
                                D_matrix(i, j) +=
                                  (flux_orientation * u_hat_n_j * k_ratio *
                                   p_hat_i_conj * JxW_face)
                                    .real();
                              }
                            else if (((current_element_trial_i == 2) ||
                                      (current_element_trial_i == 3)) &&
                                     ((current_element_trial_j == 2) ||
                                      (current_element_trial_j == 3)))
                              {
                                // (k_n/ k * p_hat_j, k_n/ k * p_hat_i_conj)
                                D_matrix(i, j) -= (k_ratio * p_hat_j * k_ratio *
                                                   p_hat_i_conj * JxW_face)
                                                    .real();
                              }
                          }
                      }
                  }
              }
          }

        // Compute the condensation matrices
        G_matrix.invert(); // G^-1

        // M4 matrix B^dagger * G^-1
        B_matrix.Tmmult(M4_matrix, G_matrix);

        // M5 matrix B_hat^dagger * G^-1
        B_hat_matrix.Tmmult(M5_matrix, G_matrix);

        // M1 matrix B^dagger * G^-1 * B
        M4_matrix.mmult(M1_matrix, B_matrix);

        // M1 matrix inverse
        M1_matrix.invert();

        // M2 matrix B^dagger * G^-1 * B_hat
        M4_matrix.mmult(M2_matrix, B_hat_matrix);

        // M3 matrix B_hat^dagger * G^-1 * B_hat - D
        M5_matrix.mmult(M3_matrix, B_hat_matrix);
        M3_matrix.add(-1.0, D_matrix);

        if (solve_interior)
          { // Solve the interior problem

            // Get the solution vector
            cell_skeleton->get_dof_values(solution_skeleton,
                                          cell_skeleton_solution);

            // Solve the interior problem
            M2_matrix.vmult(tmp_vector, cell_skeleton_solution);
            M4_matrix.vmult(cell_interior_rhs, l_vector);
            cell_interior_rhs -= tmp_vector;
            M1_matrix.vmult(cell_interior_solution, cell_interior_rhs);

            // Map the interior solution to the global solution
            cell->distribute_local_to_global(cell_interior_solution,
                                             solution_interior);
          }
        else
          { // Send the local matrices to the global matrix

            // Cell matrix M3 - M2_dagger * M1_inv * M2
            M2_matrix.Tmmult(tmp_matrix, M1_matrix);
            tmp_matrix.mmult(tmp_matrix2, M2_matrix);
            tmp_matrix2.add(-1.0, M3_matrix);
            tmp_matrix2 *= -1.0;
            cell_matrix = tmp_matrix2; // LAPACK to full matrix

            // Cell rhs (M5-M2_dagger * M1_inv M4)l - g
            tmp_matrix.mmult(tmp_matrix3, M4_matrix);
            M5_matrix.add(-1.0, tmp_matrix3);
            M5_matrix.vmult(cell_skeleton_rhs, l_vector);
            cell_skeleton_rhs -= g_vector;

            // Map to global matrix
            cell_skeleton->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_skeleton_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
      }
  }

  // @sect3{DPG::solve}
  template <int dim>
  void DPGHelmholtz<dim>::solve_skeleton()
  {
    std::cout << std::endl << "Solving the DPG system..." << std::endl;

    // Iterative solver
    SolverControl solver_control(1000000, 1e-10 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix,
                 solution_skeleton,
                 system_rhs,
                 PreconditionIdentity());
    constraints.distribute(solution_skeleton);

    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence. \n"
              << std::endl;

    error_table.add_value("n_iter", solver_control.last_step());
  }

  // @sect3{DPG::output_results}
  template <int dim>
  void DPGHelmholtz<dim>::output_results(const unsigned int cycle)
  {
    // Output cell data
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_trial_interior);

    // Organize the solution output
    std::vector<std::string> solution_interior_names(dim, "velocity_real");
    for (unsigned int i = 0; i < dim; ++i)
      {
        solution_interior_names.emplace_back("velocity_imag");
      }
    solution_interior_names.emplace_back("pressure_real");
    solution_interior_names.emplace_back("pressure_imag");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    // Output the solution
    data_out.add_data_vector(solution_interior,
                             solution_interior_names,
                             DataOut<dim>::type_automatic,
                             data_component_interpretation);

    data_out.build_patches(fe_trial_interior.degree);

    std::ofstream output("solution_planewave_square-" + std::to_string(cycle) +
                         ".vtk");
    data_out.write_vtk(output);

    // Output face data
    DataOutFaces<dim> data_out_faces(false);
    data_out_faces.attach_dof_handler(dof_handler_trial_skeleton);

    std::vector<std::string> solution_skeleton_names(1, "velocity_hat_real");
    solution_skeleton_names.emplace_back("velocity_hat_imag");
    solution_skeleton_names.emplace_back("pressure_hat_real");
    solution_skeleton_names.emplace_back("pressure_hat_imag");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_skeleton(
        4, DataComponentInterpretation::component_is_scalar);

    data_out_faces.add_data_vector(solution_skeleton,
                                   solution_skeleton_names,
                                   DataOutFaces<dim>::type_automatic,
                                   data_component_interpretation_skeleton);

    data_out_faces.build_patches(fe_trial_skeleton.degree);
    std::ofstream output_face("solution-face_planewave_square-" +
                              std::to_string(cycle) + ".vtk");
    data_out_faces.write_vtk(output_face);
  }

  // @sect3{DPG::calculate_error}
  template <int dim>
  void DPGHelmholtz<dim>::calculate_L2_error()
  {
    QGauss<dim>           quadrature_formula(fe_test.degree + 1);
    FEValues<dim>         fe_values_trial_interior(fe_trial_interior,
                                           quadrature_formula,
                                           update_values |
                                             update_quadrature_points |
                                             update_JxW_values);
    const QGauss<dim - 1> face_quadrature_formula(fe_test.degree + 1);
    FEFaceValues<dim>     fe_values_trial_skeleton(fe_trial_skeleton,
                                               face_quadrature_formula,
                                               update_values |
                                                 update_quadrature_points |
                                                 update_normal_vectors |
                                                 update_JxW_values);

    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // Create a variable to store the integration result
    double L2_error_p_real     = 0;
    double L2_error_p_imag     = 0;
    double L2_error_p_hat_real = 0;
    double L2_error_p_hat_imag = 0;
    double L2_error_u_real     = 0;
    double L2_error_u_imag     = 0;
    double L2_error_u_hat_real = 0;
    double L2_error_u_hat_imag = 0;
    double mesh_skeleton_area  = 0;

    // Create a variable to store the analytical solution scalar product with
    // normal
    double u_hat_n_analytical_real = 0.;
    double u_hat_n_analytical_imag = 0.;

    // Create an std::vector which will contain all the interpolated
    // values at the quadrature points
    std::vector<Tensor<1, dim>> local_u_values_real(n_q_points);     // u'
    std::vector<Tensor<1, dim>> local_u_values_imag(n_q_points);     // u''
    std::vector<double>         local_field_values_real(n_q_points); // p'
    std::vector<double>         local_field_values_imag(n_q_points); // p''
    std::vector<double> local_face_u_values_real(n_face_q_points);   // u_n_hat'
    std::vector<double> local_face_u_values_imag(n_face_q_points); // u_n_hat''
    std::vector<double> local_face_field_values_real(n_face_q_points); // p_hat'
    std::vector<double> local_face_field_values_imag(
      n_face_q_points); // p_hat''

    // Create the functions for the analytical solution
    AnalyticalSolution_p_real<dim> analytical_solution_p_real(wavenumber,
                                                              theta);
    AnalyticalSolution_p_imag<dim> analytical_solution_p_imag(wavenumber,
                                                              theta);
    AnalyticalSolution_u_real<dim> analytical_solution_u_real(wavenumber,
                                                              theta);
    AnalyticalSolution_u_imag<dim> analytical_solution_u_imag(wavenumber,
                                                              theta);

    // Now we loop over all the cells
    for (const auto &cell : dof_handler_trial_interior.active_cell_iterators())
      {
        // Extract local solution
        fe_values_trial_interior.reinit(cell);
        const typename DoFHandler<dim>::active_cell_iterator cell_skeleton =
          cell->as_dof_handler_iterator(dof_handler_trial_skeleton);

        fe_values_trial_interior[extractor_u_real].get_function_values(
          solution_interior, local_u_values_real);
        fe_values_trial_interior[extractor_u_imag].get_function_values(
          solution_interior, local_u_values_imag);
        fe_values_trial_interior[extractor_p_real].get_function_values(
          solution_interior, local_field_values_real);
        fe_values_trial_interior[extractor_p_imag].get_function_values(
          solution_interior, local_field_values_imag);

        // Compute the L2 error
        const auto &quadrature_points =
          fe_values_trial_interior.get_quadrature_points();

        // Loop over all quadrature points of each cell
        for (const unsigned int q_index :
             fe_values_trial_interior.quadrature_point_indices())
          {
            const double JxW      = fe_values_trial_interior.JxW(q_index);
            const auto  &position = quadrature_points[q_index];

            // Calculate the L2 error for u
            for (unsigned int i = 0; i < dim; ++i)
              {
                L2_error_u_real +=
                  pow((local_u_values_real[q_index][i] -
                       analytical_solution_u_real.value(position, i)),
                      2) *
                  JxW;
                L2_error_u_imag +=
                  pow((local_u_values_imag[q_index][i] -
                       analytical_solution_u_imag.value(position, i)),
                      2) *
                  JxW;
              }
            // Calculate the L2 error for p
            L2_error_p_real +=
              pow((local_field_values_real[q_index] -
                   analytical_solution_p_real.value(position, 0)),
                  2) *
              JxW;

            L2_error_p_imag +=
              pow((local_field_values_imag[q_index] -
                   analytical_solution_p_imag.value(position, 0)),
                  2) *
              JxW;
          }

        // Loop over all face
        for (const auto &face : cell->face_iterators())
          {
            // Reinitialization
            fe_values_trial_skeleton.reinit(cell_skeleton, face);
            const auto face_no = cell_skeleton->face_iterator_to_index(face);

            // Extract local solution
            fe_values_trial_skeleton[extractor_u_hat_real].get_function_values(
              solution_skeleton, local_face_u_values_real);
            fe_values_trial_skeleton[extractor_u_hat_imag].get_function_values(
              solution_skeleton, local_face_u_values_imag);
            fe_values_trial_skeleton[extractor_p_hat_real].get_function_values(
              solution_skeleton, local_face_field_values_real);
            fe_values_trial_skeleton[extractor_p_hat_imag].get_function_values(
              solution_skeleton, local_face_field_values_imag);

            // Compute the L2 error
            const auto &quadrature_points =
              fe_values_trial_skeleton.get_quadrature_points();

            for (const unsigned int &q_index :
                 fe_values_trial_skeleton.quadrature_point_indices())
              {
                const double JxW      = fe_values_trial_skeleton.JxW(q_index);
                const auto  &position = quadrature_points[q_index];
                const auto  &normal =
                  fe_values_trial_skeleton.normal_vector(q_index);

                // Get the neighbor cell and current cell id
                int neighbor_cell_id = -1;
                if (face->at_boundary())
                  {
                    neighbor_cell_id = INT_MAX;
                  }
                else
                  {
                    neighbor_cell_id = cell->neighbor(face_no)->index();
                  }
                const auto current_cell_id = cell->index();

                // Only calculate the error for one of the cells communicating
                // faces
                if (neighbor_cell_id < current_cell_id)
                  {
                    continue;
                  }

                // Calculate the L2 error for u_n_hat
                u_hat_n_analytical_real = 0.;
                u_hat_n_analytical_imag = 0.;
                for (unsigned int i = 0; i < dim; ++i)
                  {
                    u_hat_n_analytical_real +=
                      normal[i] * analytical_solution_u_real.value(position, i);
                    u_hat_n_analytical_imag +=
                      normal[i] * analytical_solution_u_imag.value(position, i);
                  }

                L2_error_u_hat_real +=
                  pow(abs(local_face_u_values_real[q_index]) -
                        abs(u_hat_n_analytical_real),
                      2) *
                  JxW;
                L2_error_u_hat_imag +=
                  pow(abs(local_face_u_values_imag[q_index]) -
                        abs(u_hat_n_analytical_imag),
                      2) *
                  JxW;

                // Calculate the L2 error for p_hat
                L2_error_p_hat_real +=
                  pow((local_face_field_values_real[q_index] -
                       analytical_solution_p_real.value(position, 0)),
                      2) *
                  JxW;
                L2_error_p_hat_imag +=
                  pow((local_face_field_values_imag[q_index] -
                       analytical_solution_p_imag.value(position, 0)),
                      2) *
                  JxW;

                mesh_skeleton_area += JxW;
              }
          }
      }

    // Normalize the error by the mesh area
    L2_error_p_hat_real /= mesh_skeleton_area;
    L2_error_p_hat_imag /= mesh_skeleton_area;
    L2_error_u_hat_real /= mesh_skeleton_area;
    L2_error_u_hat_imag /= mesh_skeleton_area;

    std::cout << "L2 velocity real part error is : "
              << std::sqrt(L2_error_u_real) << std::endl;
    std::cout << "L2 velocity imag part error is : "
              << std::sqrt(L2_error_u_imag) << std::endl;
    std::cout << "L2 pressure real part error is : "
              << std::sqrt(L2_error_p_real) << std::endl;
    std::cout << "L2 pressure imag part error is : "
              << std::sqrt(L2_error_p_imag) << std::endl;
    std::cout << "L2 velocity skeleton real part error is : "
              << std::sqrt(L2_error_u_hat_real) << std::endl;
    std::cout << "L2 velocity skeleton imag part error is : "
              << std::sqrt(L2_error_u_hat_imag) << std::endl;
    std::cout << "L2 pressure skeleton real part error is : "
              << std::sqrt(L2_error_p_hat_real) << std::endl;
    std::cout << "L2 presssure skeleton imag part error is : "
              << std::sqrt(L2_error_p_hat_imag) << std::endl;

    // Store the errors in the error table
    error_table.add_value("eL2_u_r", std::sqrt(L2_error_u_real));
    error_table.add_value("eL2_u_i", std::sqrt(L2_error_u_imag));
    error_table.add_value("eL2_p_r", std::sqrt(L2_error_p_real));
    error_table.add_value("eL2_p_i", std::sqrt(L2_error_p_imag));
    error_table.add_value("eL2_u_hat_r", std::sqrt(L2_error_u_hat_real));
    error_table.add_value("eL2_u_hat_i", std::sqrt(L2_error_u_hat_imag));
    error_table.add_value("eL2_p_hat_r", std::sqrt(L2_error_p_hat_real));
    error_table.add_value("eL2_p_hat_i", std::sqrt(L2_error_p_hat_imag));
  }

  // @sect3{DPG::refine_grid}
  template <int dim>
  void DPGHelmholtz<dim>::refine_grid(const unsigned int cycle)
  {
    if (cycle == 0)
      {
        const Point<dim> p1{0., 0.};
        const Point<dim> p2{1., 1.};

        std::vector<unsigned int> repetitions({2, 2});
        GridGenerator::subdivided_hyper_rectangle(
          triangulation, repetitions, p1, p2, true);
        triangulation.refine_global(0);
      }
    else
      {
        triangulation.refine_global();
      }

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    error_table.add_value("cycle", cycle);
    error_table.add_value("n_cells", triangulation.n_active_cells());
    error_table.add_value("cell_size",
                          GridTools::maximal_cell_diameter<dim>(triangulation));
  }

  // @sect3{DPG::run}
  template <int dim>
  void DPGHelmholtz<dim>::run()
  {
    for (unsigned int cycle = 0; cycle < 4; ++cycle)
      {
        std::cout << "===========================================" << std::endl
                  << "Cycle " << cycle << ':' << std::endl;

        refine_grid(cycle);
        setup_system();
        assemble_system(false);
        solve_skeleton();
        assemble_system(true); // Solve the interior problem
        calculate_L2_error();
        output_results(cycle);
      }

    // Evaluate convergence rates of interest
    error_table.evaluate_convergence_rates(
      "eL2_u_r", "n_cells", ConvergenceTable::reduction_rate_log2);
    error_table.evaluate_convergence_rates(
      "eL2_u_i", "n_cells", ConvergenceTable::reduction_rate_log2);
    error_table.evaluate_convergence_rates(
      "eL2_p_r", "n_cells", ConvergenceTable::reduction_rate_log2);
    error_table.evaluate_convergence_rates(
      "eL2_p_i", "n_cells", ConvergenceTable::reduction_rate_log2);
    error_table.evaluate_convergence_rates(
      "eL2_u_hat_r", "n_cells", ConvergenceTable::reduction_rate_log2);
    error_table.evaluate_convergence_rates(
      "eL2_u_hat_i", "n_cells", ConvergenceTable::reduction_rate_log2);
    error_table.evaluate_convergence_rates(
      "eL2_p_hat_r", "n_cells", ConvergenceTable::reduction_rate_log2);
    error_table.evaluate_convergence_rates(
      "eL2_p_hat_i", "n_cells", ConvergenceTable::reduction_rate_log2);

    std::cout << "===========================================" << std::endl;
    std::cout << "Convergence table:" << std::endl;
    error_table.write_text(std::cout);
  }
} // end of namespace Step100

// @sect3{The <code>main</code> function}

// This is the main function of the program.
int main()
{
  const unsigned int dim = 2;

  try
    {
      int degree       = 1;
      int delta_degree = 1;

      std::cout << "===========================================" << std::endl
                << "Trial order: " << degree << std::endl
                << "Test order: " << delta_degree + degree << std::endl
                << "===========================================" << std::endl
                << std::endl;

      double wavenumber = 2 * 2. * M_PI; // N oscillations times 2 pi
      double theta      = M_PI / 4.;     // Angle of incidence in radians

      Step100::DPGHelmholtz<dim> dpg_poisson(degree,
                                             delta_degree,
                                             wavenumber,
                                             theta);

      dpg_poisson.run();

      std::cout << std::endl;
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
