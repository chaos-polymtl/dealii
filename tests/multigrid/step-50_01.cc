/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2016 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------

 *
 * Testcase that created wrong level/interface matrices with certain processor
 * counts (8) due to wrong constrained dofs caused by a missing level ghost
 * cell.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
}

#include <iostream>
#include <sstream>

namespace Step50
{
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const unsigned int deg);
    void
    run();

  private:
    void
    setup_system();
    void
    assemble_system();
    void
    assemble_multigrid();
    void
    solve();
    void
    refine_grid();
    void
    output_results(const unsigned int cycle) const;

    parallel::distributed::Triangulation<dim> triangulation;
    FE_Q<dim>                                 fe;
    DoFHandler<dim>                           mg_dof_handler;

    using matrix_t = LA::MPI::SparseMatrix;
    using vector_t = LA::MPI::Vector;

    matrix_t system_matrix;

    IndexSet locally_relevant_set;

    AffineConstraints<double> constraints;

    vector_t solution;
    vector_t system_rhs;

    const unsigned int degree;

    MGLevelObject<matrix_t> mg_matrices;
    MGLevelObject<matrix_t> mg_interface_matrices;

    MGConstrainedDoFs mg_constrained_dofs;
    int               K;
  };

  template <int dim>
  class Coefficient : public Function<dim>
  {
  public:
    Coefficient(int K)
      : Function<dim>()
      , K(K)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double>           &values,
               const unsigned int             component = 0) const;

  private:
    int K;
  };



  template <int dim>
  double
  Coefficient<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    return 1.0;

    double r = 1.0;
    for (int d = 0; d < dim; ++d)
      r *= (fmod(p[d] * K, 2.0) < 1.0) ? 1.0 : -1.0;

    return r < 0.0 ? 1.0 : 1e7;
  }



  template <int dim>
  void
  Coefficient<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int             component) const
  {
    const unsigned int n_points = points.size();

    Assert(values.size() == n_points,
           ExcDimensionMismatch(values.size(), n_points));

    Assert(component == 0, ExcIndexRange(component, 0, 1));

    for (unsigned int i = 0; i < n_points; ++i)
      values[i] = Coefficient<dim>::value(points[i], component);
  }


  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(const unsigned int degree)
    : triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
    , fe(degree)
    , mg_dof_handler(triangulation)
    , degree(degree)
    , K(4)
  {}


  template <int dim>
  void
  LaplaceProblem<dim>::setup_system()
  {
    mg_dof_handler.distribute_dofs(fe);
    mg_dof_handler.distribute_mg_dofs();

    locally_relevant_set =
      DoFTools::extract_locally_relevant_dofs(mg_dof_handler);

    solution.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    system_rhs.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    constraints.reinit(mg_dof_handler.locally_owned_dofs(),
                       locally_relevant_set);
    DoFTools::make_hanging_node_constraints(mg_dof_handler, constraints);

    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary;
    Functions::ConstantFunction<dim> homogeneous_dirichlet_bc(0.0);
    dirichlet_boundary[0] = &homogeneous_dirichlet_bc;
    VectorTools::interpolate_boundary_values(mg_dof_handler,
                                             dirichlet_boundary,
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(mg_dof_handler.n_dofs(),
                               mg_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(mg_dof_handler, dsp, constraints);
    system_matrix.reinit(mg_dof_handler.locally_owned_dofs(),
                         dsp,
                         MPI_COMM_WORLD,
                         true);


    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(mg_dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(mg_dof_handler, {0});

    const unsigned int n_levels = triangulation.n_global_levels();

    mg_interface_matrices.resize(0, n_levels - 1);
    mg_interface_matrices.clear_elements();
    mg_matrices.resize(0, n_levels - 1);
    mg_matrices.clear_elements();

    for (unsigned int level = 0; level < n_levels; ++level)
      {
        DynamicSparsityPattern dsp(mg_dof_handler.n_dofs(level),
                                   mg_dof_handler.n_dofs(level));
        MGTools::make_sparsity_pattern(mg_dof_handler, dsp, level);

        mg_matrices[level].reinit(mg_dof_handler.locally_owned_mg_dofs(level),
                                  mg_dof_handler.locally_owned_mg_dofs(level),
                                  dsp,
                                  MPI_COMM_WORLD,
                                  true);

        mg_interface_matrices[level].reinit(
          mg_dof_handler.locally_owned_mg_dofs(level),
          mg_dof_handler.locally_owned_mg_dofs(level),
          dsp,
          MPI_COMM_WORLD,
          true);
      }
  }


  template <int dim>
  void
  LaplaceProblem<dim>::assemble_system()
  {
    const QGauss<dim> quadrature_formula(degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const Coefficient<dim> coefficient(K);
    std::vector<double>    coefficient_values(n_q_points);

    typename DoFHandler<dim>::active_cell_iterator cell = mg_dof_handler
                                                            .begin_active(),
                                                   endc = mg_dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs    = 0;

          fe_values.reinit(cell);

          coefficient.value_list(fe_values.get_quadrature_points(),
                                 coefficient_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
                    (coefficient_values[q_point] *
                     fe_values.shape_grad(i, q_point) *
                     fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

                cell_rhs(i) += (fe_values.shape_value(i, q_point) * 1.0 *
                                fe_values.JxW(q_point));
              }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim>
  void
  LaplaceProblem<dim>::assemble_multigrid()
  {
    QGauss<dim> quadrature_formula(1 + degree);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const Coefficient<dim> coefficient(K);
    std::vector<double>    coefficient_values(n_q_points);

    std::vector<AffineConstraints<double>> boundary_constraints(
      triangulation.n_global_levels());
    AffineConstraints<double> empty_constraints;
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        const IndexSet dofset =
          DoFTools::extract_locally_relevant_level_dofs(mg_dof_handler, level);
        boundary_constraints[level].reinit(
          mg_dof_handler.locally_owned_mg_dofs(level), dofset);
        boundary_constraints[level].add_lines(
          mg_constrained_dofs.get_refinement_edge_indices(level));
        boundary_constraints[level].add_lines(
          mg_constrained_dofs.get_boundary_indices(level));

        boundary_constraints[level].close();
      }

    typename DoFHandler<dim>::cell_iterator cell = mg_dof_handler.begin(),
                                            endc = mg_dof_handler.end();

    for (; cell != endc; ++cell)
      if (cell->level_subdomain_id() == triangulation.locally_owned_subdomain())
        {
          cell_matrix = 0;
          fe_values.reinit(cell);

          coefficient.value_list(fe_values.get_quadrature_points(),
                                 coefficient_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (coefficient_values[q_point] *
                   fe_values.shape_grad(i, q_point) *
                   fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

          cell->get_mg_dof_indices(local_dof_indices);

          boundary_constraints[cell->level()].distribute_local_to_global(
            cell_matrix, local_dof_indices, mg_matrices[cell->level()]);
          const IndexSet &interface_dofs_on_level =
            mg_constrained_dofs.get_refinement_edge_indices(cell->level());
          const unsigned int lvl = cell->level();


          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              if (interface_dofs_on_level.is_element(
                    local_dof_indices[i]) // at_refinement_edge(i)
                  && !interface_dofs_on_level.is_element(
                       local_dof_indices[j]) // !at_refinement_edge(j)
                  &&
                  ((!mg_constrained_dofs.is_boundary_index(
                      lvl, local_dof_indices[i]) &&
                    !mg_constrained_dofs.is_boundary_index(
                      lvl,
                      local_dof_indices[j])) // ( !boundary(i) && !boundary(j) )
                   || (mg_constrained_dofs.is_boundary_index(
                         lvl, local_dof_indices[i]) &&
                       local_dof_indices[i] ==
                         local_dof_indices[j]) // ( boundary(i) && boundary(j)
                                               // && i==j )
                   ))
                {
                  // do nothing, so add entries to interface matrix
                }
              else
                {
                  cell_matrix(i, j) = 0;
                }


          empty_constraints.distribute_local_to_global(
            cell_matrix,
            local_dof_indices,
            mg_interface_matrices[cell->level()]);
        }

    for (unsigned int i = 0; i < triangulation.n_global_levels(); ++i)
      {
        mg_matrices[i].compress(VectorOperation::add);
        deallog << "mg_mat" << i << ' ' << mg_matrices[i].frobenius_norm()
                << std::endl;
        mg_interface_matrices[i].compress(VectorOperation::add);
        deallog << "mg_interface_mat" << i << ' '
                << mg_interface_matrices[i].frobenius_norm() << std::endl;
      }
  }


  template <int dim>
  void
  LaplaceProblem<dim>::solve()
  {
    MGTransferPrebuilt<vector_t> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(mg_dof_handler);

    matrix_t &coarse_matrix = mg_matrices[0];

    SolverControl        coarse_solver_control(1000, 1e-10, false, false);
    SolverCG<vector_t>   coarse_solver(coarse_solver_control);
    PreconditionIdentity id;
    MGCoarseGridIterativeSolver<vector_t,
                                SolverCG<vector_t>,
                                matrix_t,
                                PreconditionIdentity>
      coarse_grid_solver(coarse_solver, coarse_matrix, id);

    using Smoother = LA::MPI::PreconditionJacobi;
    MGSmootherPrecondition<matrix_t, Smoother, vector_t> mg_smoother;
    mg_smoother.initialize(mg_matrices, Smoother::AdditionalData(0.5));
    mg_smoother.set_steps(2);
    mg::Matrix<vector_t> mg_matrix(mg_matrices);
    mg::Matrix<vector_t> mg_interface_up(mg_interface_matrices);
    mg::Matrix<vector_t> mg_interface_down(mg_interface_matrices);

    Multigrid<vector_t> mg(
      mg_matrix, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);

    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG<dim, vector_t, MGTransferPrebuilt<vector_t>> preconditioner(
      mg_dof_handler, mg, mg_transfer);


    SolverControl      solver_control(500, 1e-8 * system_rhs.l2_norm(), false);
    SolverCG<vector_t> solver(solver_control);

    if (true)
      {
        LA::MPI::Vector check1, check2, check3, tmp;
        check1.reinit(mg_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
        check2 = check1;
        check3 = check1;
        tmp    = check1;

        check1 = 1.0;
        constraints.set_zero(check1);
        check3 = check1;
        for (unsigned int i = 0; i < 10; ++i)
          {
            system_matrix.vmult(tmp, check3);
            tmp *= -1.0;
            preconditioner.vmult(check2, tmp);
            check3 += check2;
          }

        deallog << "check3 iteration: " << check3.linfty_norm() << std::endl;
      }


    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    deallog << " sol: " << solution.min() << " - " << solution.max()
            << std::endl;
  }

  template <int dim>
  void
  LaplaceProblem<dim>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    LA::MPI::Vector temp_solution;
    temp_solution.reinit(locally_relevant_set, MPI_COMM_WORLD);
    temp_solution = solution;

    KellyErrorEstimator<dim>::estimate(
      mg_dof_handler,
      QGauss<dim - 1>(degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      temp_solution,
      estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, estimated_error_per_cell, 0.3, 0.0);

    triangulation.execute_coarsening_and_refinement();
  }



  template <int dim>
  void
  LaplaceProblem<dim>::output_results(const unsigned int cycle) const
  {}


  template <int dim>
  void
  LaplaceProblem<dim>::run()
  {
    for (unsigned int cycle = 0; cycle < 2; ++cycle)
      {
        deallog << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            unsigned int n_subdiv = 1;
            GridGenerator::subdivided_hyper_cube(triangulation, n_subdiv, 0, 1);
            triangulation.refine_global(2);
            if (true)
              {
                for (typename Triangulation<dim>::active_cell_iterator cell =
                       triangulation.begin_active();
                     cell != triangulation.end();
                     ++cell)
                  if (cell->is_locally_owned() && cell->center().norm() < 0.55)
                    cell->set_refine_flag();
                triangulation.execute_coarsening_and_refinement();
                for (typename Triangulation<dim>::active_cell_iterator cell =
                       triangulation.begin_active();
                     cell != triangulation.end();
                     ++cell)
                  if (cell->is_locally_owned() && cell->center().norm() > 0.3 &&
                      cell->center().norm() < 0.42)
                    cell->set_refine_flag();

                triangulation.execute_coarsening_and_refinement();

                for (typename Triangulation<dim>::active_cell_iterator cell =
                       triangulation.begin_active();
                     cell != triangulation.end();
                     ++cell)
                  if (cell->is_locally_owned() &&
                      cell->center().norm() > 0.335 &&
                      cell->center().norm() < 0.39)
                    cell->set_refine_flag();
                triangulation.execute_coarsening_and_refinement();
              }
          }
        else
          triangulation.refine_global();

        deallog << "   Number of active cells:       "
                << triangulation.n_global_active_cells() << std::endl;

        setup_system();

        deallog << "   Number of degrees of freedom: "
                << mg_dof_handler.n_dofs() << " (by level: ";
        for (unsigned int level = 0; level < triangulation.n_global_levels();
             ++level)
          deallog << mg_dof_handler.n_dofs(level)
                  << (level == triangulation.n_global_levels() - 1 ? ")" :
                                                                     ", ");
        deallog << std::endl;

        assemble_system();
        assemble_multigrid();

        solve();
      }
  }
} // namespace Step50


int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_initlog(true);

  try
    {
      using namespace Step50;

      LaplaceProblem<2> laplace_problem(1 /*degree*/);
      laplace_problem.run();
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
      throw;
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
      throw;
    }

  return 0;
}
