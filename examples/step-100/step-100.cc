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

// @sect1{Include files}

// Most of the deal.II include files have already been covered in previous
// examples and are not commented on.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_trace.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
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
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <fstream>
#include <iostream>

// @sect2{The <code>Step100</code> class}
namespace Step100
{
  using namespace dealii;

  // Create analytical solution class for kinematic pressure (p)
  template <int dim>
  class AnalyticalSolution_p : public Function<dim>
  {
  public:
    // Overload of the value function
    virtual std::complex<double> value(const Point<dim>  &p,
                                       const unsigned int component,
                                       double             wavenumber,
                                       double             theta) const;
  };

  template <int dim>
  std::complex<double> AnalyticalSolution_p<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component,
    double                              wavenumber,
    double                              theta) const
  {
    // Imaginary unit
    std::complex<double> imag(0., 1.);

    return std::exp(-imag * wavenumber *
                    (p[0] * std::cos(theta) + p[1] * std::sin(theta)));
  }

  // Create analytical solution class velcity field (u)
  template <int dim>
  class AnalyticalSolution_u : public Function<dim>
  {
  public:
    // Overload of the value function
    virtual std::complex<double> value(const Point<dim>  &p,
                                       const unsigned int component,
                                       double             wavenumber,
                                       double             theta) const;
  };

  template <int dim>
  std::complex<double>
  AnalyticalSolution_u<dim>::value(const Point<dim>  &p,
                                   const unsigned int component,
                                   double             wavenumber,
                                   double             theta) const
  {
    // Imaginary unit
    std::complex<double> imag(0., 1.);

    if (component == 0)
      return std::cos(theta) *
             std::exp(-imag * wavenumber *
                      (p[0] * std::cos(theta) + p[1] * std::sin(theta)));
    else if (component == 1)
      return std::sin(theta) *
             std::exp(-imag * wavenumber *
                      (p[0] * std::cos(theta) + p[1] * std::sin(theta)));
    else
      throw std::runtime_error(
        "Too much components for the analytical solution");
  }

  // Create the boundary value functions
  template <int dim>
  class BoundaryValues_p_real : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_p_real(const double       wavenumber,
                          const double       theta,
                          const unsigned int n_components = 1)
      : Function<dim>(n_components)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    AnalyticalSolution_p<dim> analytical_solution_p;
    double                    wavenumber;
    double                    theta;
  };

  template <int dim>
  double BoundaryValues_p_real<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    return analytical_solution_p.value(p, 0, wavenumber, theta).real();
  }

  // Create the boundary value functions
  template <int dim>
  class BoundaryValues_p_imag : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_p_imag(const double       wavenumber,
                          const double       theta,
                          const unsigned int n_components = 1)
      : Function<dim>(n_components)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    AnalyticalSolution_p<dim> analytical_solution_p;
    double                    wavenumber;
    double                    theta;
  };

  template <int dim>
  double BoundaryValues_p_imag<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    return analytical_solution_p.value(p, 0, wavenumber, theta).imag();
  }

  template <int dim>
  class BoundaryValues_u_real : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_u_real(const double       wavenumber,
                          const double       theta,
                          const unsigned int n_components = 1)
      : Function<dim>(n_components)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    AnalyticalSolution_u<dim> analytical_solution_u;
    double                    wavenumber;
    double                    theta;
  };

  template <int dim>
  double BoundaryValues_u_real<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    return -1 * analytical_solution_u.value(p, 1, wavenumber, theta).real();
  }

  template <int dim>
  class BoundaryValues_u_imag : public Function<dim>
  {
  public:
    // Constructor
    BoundaryValues_u_imag(const double       wavenumber,
                          const double       theta,
                          const unsigned int n_components = 1)
      : Function<dim>(n_components)
      , wavenumber(wavenumber)
      , theta(theta)
    {}
    virtual double value(const Point<dim>  &p,
                         const unsigned int component) const override;

  private:
    AnalyticalSolution_u<dim> analytical_solution_u;
    double                    wavenumber;
    double                    theta;
  };

  template <int dim>
  double BoundaryValues_u_imag<dim>::value(
    const Point<dim>                   &p,
    [[maybe_unused]] const unsigned int component) const
  {
    return -1 * analytical_solution_u.value(p, 1, wavenumber, theta).imag();
  }

  template <int dim>
  class DPG
  {
  public:
    DPG(const unsigned int degree,
        const unsigned int delta_degree,
        const double       wavenumber,
        double             theta);
    void run(int degree, int delta_degree);

  private:
    void setup_system();
    void assemble_system(const bool solve_interior = false);
    void solve_skeleton();
    void refine_grid(const unsigned int cycle);
    void output_results(const unsigned int cycle);
    void calculate_L2_error();
    template <typename Number>
    void output_vector_to_csv(std::string                prefix,
                              const std::vector<Number> &vector) const;

    // Data structures for solving the system
    Triangulation<dim> triangulation;

    // Components for the interior
    const FESystem<dim> fe_trial_interior;
    DoFHandler<dim>     dof_handler_trial_interior;
    Vector<double>      solution_interior;

    // Components for the skeleton
    const FESystem<dim> fe_trial_skeleton;
    DoFHandler<dim>     dof_handler_trial_skeleton;
    Vector<double>      solution_skeleton;
    Vector<double>      system_rhs_skeleton;

    SparsityPattern           sparsity_pattern_skeleton;
    SparseMatrix<double>      system_matrix_skeleton;
    AffineConstraints<double> constraints_skeleton;

    // Components for the test space
    const FESystem<dim> fe_test;
    DoFHandler<dim>     dof_handler_test;

    // Container for the L2 error convergence
    std::vector<double> error_L2_norm;
    std::vector<double> error_L2_norm_flux_real;
    std::vector<double> error_L2_norm_flux_imag;
    std::vector<double> error_L2_norm_scalar_real;
    std::vector<double> error_L2_norm_scalar_imag;
    std::vector<double> error_L2_norm_flux_hat_real;
    std::vector<double> error_L2_norm_flux_hat_imag;
    std::vector<double> error_L2_norm_scalar_hat_real;
    std::vector<double> error_L2_norm_scalar_hat_imag;
    std::vector<double> h_size;

    // Analytical solution
    AnalyticalSolution_p<dim> analytical_solution_p;
    AnalyticalSolution_u<dim> analytical_solution_u;

    // Coefficient
    double wavenumber;
    double theta;
  };

  template <int dim>
  template <typename Number>
  void DPG<dim>::output_vector_to_csv(std::string                prefix,
                                      const std::vector<Number> &vector) const
  {
    // Write the residual to a file for plotting
    std::ofstream outputFile(prefix + ".csv");

    // Check if the file opened successfully
    if (!outputFile.is_open())
      {
        std::cerr << "Error opening file!" << std::endl;
        exit(1);
      }

    // Write vector elements to the file separated by commas
    for (size_t i = 0; i < vector.size(); ++i)
      {
        outputFile << vector[i];
        // Add a comma after each element except the last one
        if (i != vector.size() - 1)
          {
            outputFile << ",";
          }
      }
    outputFile << std::endl;

    // Close the file
    outputFile.close();
  }

  // @sect3{DPG class implementation}
  // The Q elements have a degree higher than the others because their
  // numerotation start at 1 instead of 0.

  template <int dim>
  DPG<dim>::DPG(const unsigned int degree,
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
    , // (tau, tau_imag, v, v_imag)
    dof_handler_test(triangulation)
    , wavenumber(wavenumber)
    , theta(theta)
  {}

  // @sect4{DPG::setup_system}
  template <int dim>
  void DPG<dim>::setup_system()
  {
    dof_handler_trial_skeleton.distribute_dofs(fe_trial_skeleton);
    dof_handler_trial_interior.distribute_dofs(fe_trial_interior);
    dof_handler_test.distribute_dofs(fe_test);

    std::cout << "Number of degrees of freedom on the interior: "
              << dof_handler_trial_interior.n_dofs() << std::endl;

    std::cout << "Number of degrees of freedom on the skeleton: "
              << dof_handler_trial_skeleton.n_dofs() << std::endl;

    std::cout << "Number of degrees of freedom on the test space: "
              << dof_handler_test.n_dofs() << std::endl;

    std::cout << "Total number of degrees of freedom: "
              << dof_handler_trial_interior.n_dofs() +
                   dof_handler_trial_skeleton.n_dofs() +
                   dof_handler_test.n_dofs()
              << std::endl;

    constraints_skeleton.clear();

    // Define the constraints for each case
    const FEValuesExtractors::Scalar trial_face_u_real(0);
    const FEValuesExtractors::Scalar trial_face_u_imag(1);
    const FEValuesExtractors::Scalar trial_face_p_real(2);
    const FEValuesExtractors::Scalar trial_face_p_imag(3);

    IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler_trial_skeleton);
    constraints_skeleton.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler_trial_skeleton,
                                            constraints_skeleton);

    BoundaryValues_p_real<dim> p_real(wavenumber, theta, 4);
    BoundaryValues_p_imag<dim> p_imag(wavenumber, theta, 4);

    BoundaryValues_u_real<dim> u_real(wavenumber, theta, 4);
    BoundaryValues_u_imag<dim> u_imag(wavenumber, theta, 4);

    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             0,
                                             p_real,
                                             constraints_skeleton,
                                             fe_trial_skeleton.component_mask(
                                               trial_face_p_real));
    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             0,
                                             p_imag,
                                             constraints_skeleton,
                                             fe_trial_skeleton.component_mask(
                                               trial_face_p_imag));

    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             2,
                                             u_real,
                                             constraints_skeleton,
                                             fe_trial_skeleton.component_mask(
                                               trial_face_u_real));
    VectorTools::interpolate_boundary_values(dof_handler_trial_skeleton,
                                             2,
                                             u_imag,
                                             constraints_skeleton,
                                             fe_trial_skeleton.component_mask(
                                               trial_face_u_imag));

    constraints_skeleton.close();

    solution_skeleton.reinit(dof_handler_trial_skeleton.n_dofs());
    system_rhs_skeleton.reinit(dof_handler_trial_skeleton.n_dofs());
    solution_interior.reinit(dof_handler_trial_interior.n_dofs());

    DynamicSparsityPattern dsp(dof_handler_trial_skeleton.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_trial_skeleton,
                                    dsp,
                                    constraints_skeleton,
                                    false);
    sparsity_pattern_skeleton.copy_from(dsp);
    system_matrix_skeleton.reinit(sparsity_pattern_skeleton);
  };

  // @sect4{DPG::assemble_system}

  template <int dim>
  void DPG<dim>::assemble_system(const bool solve_interior)
  {
    // Define the imaginary unit
    std::complex<double> imag(0., 1.);

    // Define exctractors
    const FEValuesExtractors::Vector trial_u(0);
    const FEValuesExtractors::Vector trial_u_imag(dim);
    const FEValuesExtractors::Scalar trial_p(2 * dim);
    const FEValuesExtractors::Scalar trial_p_imag(2 * dim + 1);

    const FEValuesExtractors::Scalar trial_skeleton_u(0);
    const FEValuesExtractors::Scalar trial_skeleton_u_imag(1);
    const FEValuesExtractors::Scalar trial_skeleton_p(2);
    const FEValuesExtractors::Scalar trial_skeleton_p_imag(3);

    const FEValuesExtractors::Vector test_u(0);
    const FEValuesExtractors::Vector test_u_imag(dim);
    const FEValuesExtractors::Scalar test_p(2 * dim);
    const FEValuesExtractors::Scalar test_p_imag(2 * dim + 1);

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

    if (!solve_interior)
      {
        std::cout << "Number of dofs per cell test total: "
                  << dofs_per_cell_test << std::endl;
        std::cout << "Number of dofs per cell trial skeleton: "
                  << dofs_per_cell_trial_skeleton << std::endl;
        std::cout << "Number of dofs per cell trial interior: "
                  << dofs_per_cell_trial_interior << std::endl;
      }

    // Create the DPG local matrices
    LAPACKFullMatrix<double> G_matrix(dofs_per_cell_test, dofs_per_cell_test);
    LAPACKFullMatrix<double> B_matrix(dofs_per_cell_test,
                                      dofs_per_cell_trial_interior);
    LAPACKFullMatrix<double> B_hat_matrix(dofs_per_cell_test,
                                          dofs_per_cell_trial_skeleton);
    LAPACKFullMatrix<double> D_matrix(dofs_per_cell_trial_skeleton,
                                      dofs_per_cell_trial_skeleton);
    Vector<double>           g_vector(dofs_per_cell_trial_skeleton);
    Vector<double>           l_vector(dofs_per_cell_test);

    // Create the condensation matrices
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

    LAPACKFullMatrix<double> tmp_matrix(dofs_per_cell_trial_skeleton,
                                        dofs_per_cell_trial_interior);

    LAPACKFullMatrix<double> tmp_matrix2(dofs_per_cell_trial_skeleton,
                                         dofs_per_cell_trial_skeleton);

    LAPACKFullMatrix<double> tmp_matrix3(dofs_per_cell_trial_skeleton,
                                         dofs_per_cell_test);

    Vector<double> tmp_vector(dofs_per_cell_trial_interior);

    // Create the local resulting matrices
    FullMatrix<double> cell_matrix(dofs_per_cell_trial_skeleton,
                                   dofs_per_cell_trial_skeleton);
    Vector<double>     cell_interior_rhs(dofs_per_cell_trial_interior);
    Vector<double>     cell_skeleton_rhs(dofs_per_cell_trial_skeleton);
    Vector<double>     cell_interior_solution(dofs_per_cell_trial_interior);
    Vector<double>     cell_skeleton_solution(dofs_per_cell_trial_skeleton);

    // Create the dofs indices mapping container
    std::vector<types::global_dof_index> local_dof_indices(
      dofs_per_cell_trial_skeleton);

    // Loop over all cells
    for (const auto &cell : dof_handler_test.active_cell_iterators())
      {
        // Reinitialization
        fe_values_test.reinit(cell);

        const typename DoFHandler<dim>::active_cell_iterator cell_skeleton =
          cell->as_dof_handler_iterator(dof_handler_trial_skeleton);

        const typename DoFHandler<dim>::active_cell_iterator cell_interior =
          cell->as_dof_handler_iterator(dof_handler_trial_interior);
        fe_values_trial_interior.reinit(cell_interior);

        // Reinitialization of the matrices
        G_matrix.reinit(dofs_per_cell_test, dofs_per_cell_test);
        B_matrix.reinit(dofs_per_cell_test, dofs_per_cell_trial_interior);
        B_hat_matrix.reinit(dofs_per_cell_test, dofs_per_cell_trial_skeleton);
        D_matrix.reinit(dofs_per_cell_trial_skeleton,
                        dofs_per_cell_trial_skeleton);
        g_vector.reinit(dofs_per_cell_trial_skeleton);
        l_vector.reinit(dofs_per_cell_test);

        M1_matrix.reinit(dofs_per_cell_trial_interior,
                         dofs_per_cell_trial_interior);
        M2_matrix.reinit(dofs_per_cell_trial_interior,
                         dofs_per_cell_trial_skeleton);
        M3_matrix.reinit(dofs_per_cell_trial_skeleton,
                         dofs_per_cell_trial_skeleton);
        M4_matrix.reinit(dofs_per_cell_trial_interior, dofs_per_cell_test);
        M5_matrix.reinit(dofs_per_cell_trial_skeleton, dofs_per_cell_test);

        // Loop over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const std::complex<double> iomega      = imag * wavenumber;
            const std::complex<double> conj_iomega = conj(iomega);
            const double &JxW = fe_values_trial_interior.JxW(q_point);

            // Loop over test space dofs
            for (const auto i : fe_values_test.dof_indices())
              {
                // Define the necessary complex test basis functions
                const auto tau_i_conj =
                  fe_values_test[test_u].value(i, q_point) -
                  imag * fe_values_test[test_u_imag].value(i, q_point);

                const auto tau_i_div_conj =
                  fe_values_test[test_u].divergence(i, q_point) -
                  imag * fe_values_test[test_u_imag].divergence(i, q_point);

                const auto v_i_conj =
                  fe_values_test[test_p].value(i, q_point) -
                  imag * fe_values_test[test_p_imag].value(i, q_point);

                const auto v_i_grad_conj =
                  fe_values_test[test_p].gradient(i, q_point) -
                  imag * fe_values_test[test_p_imag].gradient(i, q_point);

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
                    const auto tau_j =
                      fe_values_test[test_u].value(j, q_point) +
                      imag * fe_values_test[test_u_imag].value(j, q_point);

                    const auto tau_j_div =
                      fe_values_test[test_u].divergence(j, q_point) +
                      imag * fe_values_test[test_u_imag].divergence(j, q_point);

                    const auto v_j =
                      fe_values_test[test_p].value(j, q_point) +
                      imag * fe_values_test[test_p_imag].value(j, q_point);

                    const auto v_j_grad =
                      fe_values_test[test_p].gradient(j, q_point) +
                      imag * fe_values_test[test_p_imag].gradient(j, q_point);

                    // Get the information on witch element the dof is
                    const unsigned int current_element_test_j =
                      fe_test.system_to_base_index(j).first.first;

                    // If both Raviart-thomas element
                    if (((current_element_test_i == 0) ||
                         (current_element_test_i == 1)) &&
                        ((current_element_test_j == 0) ||
                         (current_element_test_j == 1)))
                      {
                        // (tau,tau*) + (div(tau),div(tau)*) + (i omega tau, (i
                        // omega tau)*)
                        G_matrix(i, j) +=
                          (((tau_j * tau_i_conj) +
                            (tau_j_div * tau_i_div_conj) +
                            (iomega * tau_j * conj_iomega * tau_i_conj)) *
                           JxW)
                            .real();
                      }
                    // If in i Raviart-thomas element and j in Q element
                    else if (((current_element_test_i == 0) ||
                              (current_element_test_i == 1)) &&
                             ((current_element_test_j == 2) ||
                              (current_element_test_j == 3)))
                      {
                        // (grad(v), (i omega tau)*) + (i omega v, div(tau) *)
                        G_matrix(i, j) +=
                          (((v_j_grad * conj_iomega * tau_i_conj) +
                            (iomega * v_j * tau_i_div_conj)) *
                           JxW)
                            .real();
                      }
                    // If in i Q element and j in Raviart-thomas element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_test_j == 0) ||
                              (current_element_test_j == 1)))
                      {
                        // ( i omega tau , grad(v)*) + (div(tau), (i omega v) *)
                        G_matrix(i, j) +=
                          (((iomega * tau_j * v_i_grad_conj) +
                            (tau_j_div * conj_iomega * v_i_conj)) *
                           JxW)
                            .real();
                      }
                    // If both Q element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_test_j == 2) ||
                              (current_element_test_j == 3)))
                      {
                        // (v,v*) + (grad(v),grad(v)*) + (i omega v, (i omega
                        // v)*)
                        G_matrix(i, j) +=
                          (((v_j * v_i_conj) + (v_j_grad * v_i_grad_conj) +
                            (iomega * v_j * conj_iomega * v_i_conj)) *
                           JxW)
                            .real();
                      }
                  }

                // Loop over trial space dofs
                for (const auto j : fe_values_trial_interior.dof_indices())
                  {
                    // Create the trial basis functions
                    const auto u_j =
                      fe_values_trial_interior[trial_u].value(j, q_point) +
                      imag *
                        fe_values_trial_interior[trial_u_imag].value(j,
                                                                     q_point);

                    const auto p_j =
                      fe_values_trial_interior[trial_p].value(j, q_point) +
                      imag *
                        fe_values_trial_interior[trial_p_imag].value(j,
                                                                     q_point);

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
                        // (i omega u, tau*)
                        B_matrix(i, j) +=
                          ((iomega * u_j * tau_i_conj) * JxW).real();
                      }
                    // If in Raviart-thomas element and DGQ element
                    else if (((current_element_test_i == 0) ||
                              (current_element_test_i == 1)) &&
                             ((current_element_trial_j == 2) ||
                              (current_element_trial_j == 3)))
                      {
                        // -(p,div(tau)*)
                        B_matrix(i, j) -= ((p_j * tau_i_div_conj) * JxW).real();
                      }

                    // If in Q element and DGQ^dim element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_trial_j == 0) ||
                              (current_element_trial_j == 1)))
                      {
                        // -(u,grad(v)*)
                        B_matrix(i, j) -= ((u_j * v_i_grad_conj) * JxW).real();
                      }
                    // If in Q element and DGQ element
                    else if (((current_element_test_i == 2) ||
                              (current_element_test_i == 3)) &&
                             ((current_element_trial_j == 2) ||
                              (current_element_trial_j == 3)))
                      {
                        // (i omega p, v*)
                        B_matrix(i, j) +=
                          ((iomega * p_j * v_i_conj) * JxW).real();
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
                    const auto tau_n_i_conj =
                      normal *
                      (fe_face_values_test[test_u].value(i, q_point) -
                       imag *
                         fe_face_values_test[test_u_imag].value(i, q_point));

                    const auto v_i_conj =
                      fe_face_values_test[test_p].value(i, q_point) -
                      imag * fe_face_values_test[test_p_imag].value(i, q_point);

                    // Get the information to map the index to the right shape
                    // function
                    const unsigned int current_element_test_i =
                      fe_test.system_to_base_index(i).first.first;

                    // Loop over trial space dofs
                    for (const auto j : fe_values_trial_skeleton.dof_indices())
                      {
                        // Create the face trial basis functions
                        const auto u_hat_n_j =
                          fe_values_trial_skeleton[trial_skeleton_u].value(
                            j, q_point) +
                          imag * fe_values_trial_skeleton[trial_skeleton_u_imag]
                                   .value(j, q_point);

                        const auto p_hat_j =
                          fe_values_trial_skeleton[trial_skeleton_p].value(
                            j, q_point) +
                          imag * fe_values_trial_skeleton[trial_skeleton_p_imag]
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
                              ((p_hat_j * tau_n_i_conj) * JxW_face).real();
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

                            // (u_hat_n, v*)
                            B_hat_matrix(i, j) +=
                              (flux_orientation * u_hat_n_j * v_i_conj *
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
                        const auto tau_n_i_conj =
                          normal *
                          (fe_face_values_test[test_u].value(i, q_point) -
                           imag *
                             fe_face_values_test[test_u_imag].value(i,
                                                                    q_point));

                        const auto v_i_conj =
                          fe_face_values_test[test_p].value(i, q_point) -
                          imag *
                            fe_face_values_test[test_p_imag].value(i, q_point);

                        const unsigned int current_element_test_i =
                          fe_test.system_to_base_index(i).first.first;

                        for (const auto j : fe_face_values_test.dof_indices())
                          {
                            // Create the face test basis functions
                            const auto tau_n_j =
                              normal *
                              (fe_face_values_test[test_u].value(j, q_point) +
                               imag * fe_face_values_test[test_u_imag].value(
                                        j, q_point));

                            const auto v_j =
                              fe_face_values_test[test_p].value(j, q_point) +
                              imag *
                                fe_face_values_test[test_p_imag].value(j,
                                                                       q_point);

                            const unsigned int current_element_test_j =
                              fe_test.system_to_base_index(j).first.first;

                            if (((current_element_test_i == 0) ||
                                 (current_element_test_i == 1)) &&
                                ((current_element_test_j == 0) ||
                                 (current_element_test_j == 1)))
                              {
                                // -(tau_n_j, tau_n_i*)
                                G_matrix(i, j) -=
                                  (tau_n_j * tau_n_i_conj * JxW_face).real();
                              }
                            else if (((current_element_test_i == 0) ||
                                      (current_element_test_i == 1)) &&
                                     ((current_element_test_j == 2) ||
                                      (current_element_test_j == 3)))
                              {
                                // (k_n/ k * v_j, tau_n_i*)
                                G_matrix(i, j) +=
                                  (k_ratio * v_j * tau_n_i_conj * JxW_face)
                                    .real();
                              }
                            else if (((current_element_test_i == 2) ||
                                      (current_element_test_i == 3)) &&
                                     ((current_element_test_j == 0) ||
                                      (current_element_test_j == 1)))
                              {
                                // (tau_n_j, k_n/ k * v_i_conj)
                                G_matrix(i, j) +=
                                  (tau_n_j * k_ratio * v_i_conj * JxW_face)
                                    .real();
                              }
                            else if (((current_element_test_i == 2) ||
                                      (current_element_test_i == 3)) &&
                                     ((current_element_test_j == 2) ||
                                      (current_element_test_j == 3)))
                              {
                                // (k_n/ k * v_j, k_n/ k * v_i_conj)
                                G_matrix(i, j) -= (k_ratio * v_j * k_ratio *
                                                   v_i_conj * JxW_face)
                                                    .real();
                              }
                          }
                      }

                    // Update the g_vector and D_matrix
                    for (const auto i : fe_values_trial_skeleton.dof_indices())
                      {
                        // Create the face trial basis functions
                        const auto u_hat_n_i_conj =
                          fe_values_trial_skeleton[trial_skeleton_u].value(
                            i, q_point) -
                          imag * fe_values_trial_skeleton[trial_skeleton_u_imag]
                                   .value(i, q_point);

                        const auto p_hat_i_conj =
                          fe_values_trial_skeleton[trial_skeleton_p].value(
                            i, q_point) -
                          imag * fe_values_trial_skeleton[trial_skeleton_p_imag]
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
                              fe_values_trial_skeleton[trial_skeleton_u].value(
                                j, q_point) +
                              imag *
                                fe_values_trial_skeleton[trial_skeleton_u_imag]
                                  .value(j, q_point);

                            const auto p_hat_j =
                              fe_values_trial_skeleton[trial_skeleton_p].value(
                                j, q_point) +
                              imag *
                                fe_values_trial_skeleton[trial_skeleton_p_imag]
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
            cell_interior->distribute_local_to_global(cell_interior_solution,
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
            constraints_skeleton.distribute_local_to_global(
              cell_matrix,
              cell_skeleton_rhs,
              local_dof_indices,
              system_matrix_skeleton,
              system_rhs_skeleton);
          }
      }
  }

  // @sect4{DPG::solve}
  template <int dim>
  void DPG<dim>::solve_skeleton()
  {
    // Iterative solver
    SolverControl            solver_control(1000000,
                                 1e-10 * system_rhs_skeleton.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix_skeleton,
                 solution_skeleton,
                 system_rhs_skeleton,
                 PreconditionIdentity());
    constraints_skeleton.distribute(solution_skeleton);

    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;
  }

  // @sect4{DPG::output_results}
  template <int dim>
  void DPG<dim>::output_results(const unsigned int cycle)
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

  // @sect4{DPG::calculate_error}
  template <int dim>
  void DPG<dim>::calculate_L2_error()
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
    double L2_error_scalar_real     = 0;
    double L2_error_scalar_imag     = 0;
    double L2_error_scalar_hat_real = 0;
    double L2_error_scalar_hat_imag = 0;
    double L2_error_flux_real       = 0;
    double L2_error_flux_imag       = 0;
    double L2_error_flux_hat_real   = 0;
    double L2_error_flux_hat_imag   = 0;
    double L2_error                 = 0;
    double mesh_skeleton_area       = 0;

    // Create a variable to store the analytical solution scalar product with
    // normal
    std::complex<double> u_hat_n_analytical = 0.;

    // Create an std::vector which will contain all the interpolated
    // values at the quadrature points
    std::vector<Tensor<1, dim>> local_flux_values_real(n_q_points);  // u'
    std::vector<Tensor<1, dim>> local_flux_values_imag(n_q_points);  // u''
    std::vector<double>         local_field_values_real(n_q_points); // p'
    std::vector<double>         local_field_values_imag(n_q_points); // p''
    std::vector<double>         local_face_flux_values_real(
      n_face_q_points); // u_n_hat'
    std::vector<double> local_face_flux_values_imag(
      n_face_q_points); // u_n_hat''
    std::vector<double> local_face_field_values_real(n_face_q_points); // p_hat'
    std::vector<double> local_face_field_values_imag(
      n_face_q_points); // p_hat''

    // Define extractors
    const FEValuesExtractors::Vector trial_u(0);
    const FEValuesExtractors::Vector trial_u_imag(dim);
    const FEValuesExtractors::Scalar trial_p(2 * dim);
    const FEValuesExtractors::Scalar trial_p_imag(2 * dim + 1);
    const FEValuesExtractors::Scalar trial_face_u(0);
    const FEValuesExtractors::Scalar trial_face_u_imag(1);
    const FEValuesExtractors::Scalar trial_face_p(2);
    const FEValuesExtractors::Scalar trial_face_p_imag(3);

    // Now we loop over all the cells
    for (const auto &cell : dof_handler_trial_interior.active_cell_iterators())
      {
        // Extract local solution
        fe_values_trial_interior.reinit(cell);
        const typename DoFHandler<dim>::active_cell_iterator cell_skeleton =
          cell->as_dof_handler_iterator(dof_handler_trial_skeleton);

        fe_values_trial_interior[trial_u].get_function_values(
          solution_interior, local_flux_values_real);
        fe_values_trial_interior[trial_u_imag].get_function_values(
          solution_interior, local_flux_values_imag);
        fe_values_trial_interior[trial_p].get_function_values(
          solution_interior, local_field_values_real);
        fe_values_trial_interior[trial_p_imag].get_function_values(
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
                L2_error_flux_real +=
                  pow((local_flux_values_real[q_index][i] -
                       analytical_solution_u
                         .value(position, i, wavenumber, theta)
                         .real()),
                      2) *
                  JxW;
                L2_error_flux_imag +=
                  pow((local_flux_values_imag[q_index][i] -
                       analytical_solution_u
                         .value(position, i, wavenumber, theta)
                         .imag()),
                      2) *
                  JxW;
              }
            // Calculate the L2 error for p
            L2_error_scalar_real +=
              pow((local_field_values_real[q_index] -
                   analytical_solution_p.value(position, 0, wavenumber, theta)
                     .real()),
                  2) *
              JxW;

            L2_error_scalar_imag +=
              pow((local_field_values_imag[q_index] -
                   analytical_solution_p.value(position, 0, wavenumber, theta)
                     .imag()),
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
            fe_values_trial_skeleton[trial_face_u].get_function_values(
              solution_skeleton, local_face_flux_values_real);
            fe_values_trial_skeleton[trial_face_u_imag].get_function_values(
              solution_skeleton, local_face_flux_values_imag);
            fe_values_trial_skeleton[trial_face_p].get_function_values(
              solution_skeleton, local_face_field_values_real);
            fe_values_trial_skeleton[trial_face_p_imag].get_function_values(
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
                u_hat_n_analytical = 0.;
                for (unsigned int i = 0; i < dim; ++i)
                  {
                    u_hat_n_analytical +=
                      normal[i] * analytical_solution_u.value(position,
                                                              i,
                                                              wavenumber,
                                                              theta);
                  }

                L2_error_flux_hat_real +=
                  pow(abs(local_face_flux_values_real[q_index]) -
                        abs(u_hat_n_analytical.real()),
                      2) *
                  JxW;
                L2_error_flux_hat_imag +=
                  pow(abs(local_face_flux_values_imag[q_index]) -
                        abs(u_hat_n_analytical.imag()),
                      2) *
                  JxW;

                // Calculate the L2 error for p_hat
                L2_error_scalar_hat_real +=
                  pow((local_face_field_values_real[q_index] -
                       analytical_solution_p
                         .value(position, 0, wavenumber, theta)
                         .real()),
                      2) *
                  JxW;
                L2_error_scalar_hat_imag +=
                  pow((local_face_field_values_imag[q_index] -
                       analytical_solution_p
                         .value(position, 0, wavenumber, theta)
                         .imag()),
                      2) *
                  JxW;

                mesh_skeleton_area += JxW;
              }
          }
      }

    L2_error = L2_error_scalar_real + L2_error_scalar_imag +
               L2_error_flux_real + L2_error_flux_imag;
    L2_error_scalar_hat_real /= mesh_skeleton_area;
    L2_error_scalar_hat_imag /= mesh_skeleton_area;
    L2_error_flux_hat_real /= mesh_skeleton_area;
    L2_error_flux_hat_imag /= mesh_skeleton_area;

    std::cout << "L2 error is : " << std::sqrt(L2_error) << std::endl;
    std::cout << "L2 pressure real part error is : "
              << std::sqrt(L2_error_scalar_real) << std::endl;
    std::cout << "L2 pressure imag part error is : "
              << std::sqrt(L2_error_scalar_imag) << std::endl;
    std::cout << "L2 velocity real part error is : "
              << std::sqrt(L2_error_flux_real) << std::endl;
    std::cout << "L2 velocity imag part error is : "
              << std::sqrt(L2_error_flux_imag) << std::endl;
    std::cout << "L2 velocity skeleton real part error is : "
              << std::sqrt(L2_error_flux_hat_real) << std::endl;
    std::cout << "L2 velocity skeleton imag part error is : "
              << std::sqrt(L2_error_flux_hat_imag) << std::endl;
    std::cout << "L2 pressure skeleton real part error is : "
              << std::sqrt(L2_error_scalar_hat_real) << std::endl;
    std::cout << "L2 presssure skeleton imag part error is : "
              << std::sqrt(L2_error_scalar_hat_imag) << std::endl;

    error_L2_norm.push_back(std::sqrt(L2_error));
    error_L2_norm_flux_real.push_back(std::sqrt(L2_error_flux_real));
    error_L2_norm_flux_imag.push_back(std::sqrt(L2_error_flux_imag));
    error_L2_norm_scalar_real.push_back(std::sqrt(L2_error_scalar_real));
    error_L2_norm_scalar_imag.push_back(std::sqrt(L2_error_scalar_imag));
    error_L2_norm_flux_hat_real.push_back(std::sqrt(L2_error_flux_hat_real));
    error_L2_norm_flux_hat_imag.push_back(std::sqrt(L2_error_flux_hat_imag));
    error_L2_norm_scalar_hat_real.push_back(
      std::sqrt(L2_error_scalar_hat_real));
    error_L2_norm_scalar_hat_imag.push_back(
      std::sqrt(L2_error_scalar_hat_imag));
  }

  // @sect4{DPG::refine_grid}
  template <int dim>
  void DPG<dim>::refine_grid(const unsigned int cycle)
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

    h_size.push_back(GridTools::maximal_cell_diameter<dim>(triangulation));
  }

  // @sect4{DPG::run}
  template <int dim>
  void DPG<dim>::run(int degree, int delta_degree)
  {
    for (unsigned int cycle = 0; cycle < 8; ++cycle)
      {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        auto timer_total_start = std::chrono::high_resolution_clock::now();
        refine_grid(cycle);
        setup_system();
        auto timer_assembly_start = std::chrono::high_resolution_clock::now();
        assemble_system(false);
        auto timer_assembly_end = std::chrono::high_resolution_clock::now();
        auto timer_solve_start  = std::chrono::high_resolution_clock::now();
        solve_skeleton();
        auto timer_solve_end = std::chrono::high_resolution_clock::now();
        auto timer_solve_interior_start =
          std::chrono::high_resolution_clock::now();
        assemble_system(true); // Solve the interior problem
        auto timer_solve_interior_end =
          std::chrono::high_resolution_clock::now();
        calculate_L2_error();
        output_results(cycle);
        auto timer_total_end = std::chrono::high_resolution_clock::now();

        // Print to consol the different times
        std::cout << "----------------------------------------------------"
                  << std::endl;
        std::cout << "Total time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                       timer_total_end - timer_total_start)
                       .count()
                  << std::endl;
        std::cout << "Assembly time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                       timer_assembly_end - timer_assembly_start)
                       .count()
                  << std::endl;
        std::cout << "Solve skeleton time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                       timer_solve_end - timer_solve_start)
                       .count()
                  << std::endl;
        std::cout << "Solve interior time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                       timer_solve_interior_end - timer_solve_interior_start)
                       .count()
                  << std::endl;
        std::cout << "----------------------------------------------------"
                  << std::endl;
      }
    output_vector_to_csv("L2_error_" + std::to_string(degree) + "_" +
                           std::to_string(delta_degree),
                         error_L2_norm);
    output_vector_to_csv("L2_error_flux_real_" + std::to_string(degree) + "_" +
                           std::to_string(delta_degree),
                         error_L2_norm_flux_real);
    output_vector_to_csv("L2_error_flux_imag_" + std::to_string(degree) + "_" +
                           std::to_string(delta_degree),
                         error_L2_norm_flux_imag);
    output_vector_to_csv("L2_error_scalar_real_" + std::to_string(degree) +
                           "_" + std::to_string(delta_degree),
                         error_L2_norm_scalar_real);
    output_vector_to_csv("L2_error_scalar_imag_" + std::to_string(degree) +
                           "_" + std::to_string(delta_degree),
                         error_L2_norm_scalar_imag);
    output_vector_to_csv("L2_error_flux_hat_real_" + std::to_string(degree) +
                           "_" + std::to_string(delta_degree),
                         error_L2_norm_flux_hat_real);
    output_vector_to_csv("L2_error_flux_hat_imag_" + std::to_string(degree) +
                           "_" + std::to_string(delta_degree),
                         error_L2_norm_flux_hat_imag);
    output_vector_to_csv("L2_error_scalar_hat_real_" + std::to_string(degree) +
                           "_" + std::to_string(delta_degree),
                         error_L2_norm_scalar_hat_real);
    output_vector_to_csv("L2_error_scalar_hat_imag_" + std::to_string(degree) +
                           "_" + std::to_string(delta_degree),
                         error_L2_norm_scalar_hat_imag);
    output_vector_to_csv("h_size_" + std::to_string(degree) + "_" +
                           std::to_string(delta_degree),
                         h_size);
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

      std::cout << "Solving with order " << degree
                << " elements with a test space " << delta_degree
                << " order higher." << std::endl
                << "===========================================" << std::endl
                << std::endl;

      double wavenumber = 2 * 2. * M_PI; // N oscillations times 2 pi
      double theta      = M_PI / 4.;     // Angle of incidence in radians

      Step100::DPG<dim> dpg_poisson(degree, delta_degree, wavenumber, theta);

      dpg_poisson.run(degree, delta_degree);

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