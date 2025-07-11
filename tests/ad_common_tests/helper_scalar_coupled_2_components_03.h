// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2019 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Header file:
// Evaluation of a coupled system (tensor + scalar components)
// using a helper class

#include <deal.II/base/logstream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <iostream>

#include "../tests.h"

namespace AD = dealii::Differentiation::AD;

// Function and its derivatives
template <int dim, typename NumberType>
struct FunctionsTestTensorScalarCoupled
{
  static NumberType
  psi(const Tensor<2, dim, NumberType> &t, const NumberType &s)
  {
    return double_contract<0, 0, 1, 1>(t, t) * pow(s, 3);
  };

  static Tensor<2, dim, NumberType>
  dpsi_dt(const Tensor<2, dim, NumberType> &t, const NumberType &s)
  {
    return 2.0 * t * pow(s, 3);
  };

  static NumberType
  dpsi_ds(const Tensor<2, dim, NumberType> &t, const NumberType &s)
  {
    return 3.0 * double_contract<0, 0, 1, 1>(t, t) * pow(s, 2);
  };

  static Tensor<4, dim, NumberType>
  d2psi_dt_dt(const Tensor<2, dim, NumberType> &t, const NumberType &s)
  {
    // Non-symmetric fourth order identity tensor
    static const SymmetricTensor<2, dim, NumberType> I(
      unit_symmetric_tensor<dim, NumberType>());
    Tensor<4, dim, NumberType> II;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            II[i][j][k][l] = I[i][k] * I[j][l];

    return 2.0 * II * pow(s, 3);
  };

  static Tensor<2, dim, NumberType>
  d2psi_ds_dt(const Tensor<2, dim, NumberType> &t, const NumberType &s)
  {
    return 6.0 * t * pow(s, 2);
  };

  static Tensor<2, dim, NumberType>
  d2psi_dt_ds(const Tensor<2, dim, NumberType> &t, const NumberType &s)
  {
    return d2psi_ds_dt(t, s);
  };

  static NumberType
  d2psi_ds_ds(const Tensor<2, dim, NumberType> &t, const NumberType &s)
  {
    return 6.0 * double_contract<0, 0, 1, 1>(t, t) * pow(s, 1);
  };
};

template <int dim, typename number_t, enum AD::NumberTypes ad_type_code>
void
test_tensor_scalar_coupled()
{
  using ADHelper         = AD::ScalarFunction<dim, ad_type_code, number_t>;
  using ADNumberType     = typename ADHelper::ad_type;
  using ScalarNumberType = typename ADHelper::scalar_type;

  std::cout << "*** Test variables: Tensor + Scalar (coupled), "
            << "dim = " << Utilities::to_string(dim) << ", "
            << "Type code: " << static_cast<int>(ad_type_code) << std::endl;

  // Values computed from the AD energy function
  ScalarNumberType             psi;
  Vector<ScalarNumberType>     Dpsi;
  FullMatrix<ScalarNumberType> D2psi;

  // Function and its derivatives
  using func_ad = FunctionsTestTensorScalarCoupled<dim, ADNumberType>;

  const FEValuesExtractors::Tensor<2> t_dof(0);
  const FEValuesExtractors::Scalar    s_dof(
    Tensor<2, dim>::n_independent_components);
  const unsigned int n_AD_components =
    Tensor<2, dim>::n_independent_components + 1;
  ADHelper ad_helper(n_AD_components);
  ad_helper.set_tape_buffer_sizes(); // Increase the buffer size from the
                                     // default values

  ScalarNumberType                 s = 7.5;
  Tensor<2, dim, ScalarNumberType> t =
    unit_symmetric_tensor<dim, ScalarNumberType>();
  for (unsigned int i = 0; i < t.n_independent_components; ++i)
    t[t.unrolled_to_component_indices(i)] += 0.18 * (i + 0.12);

  const int  tape_no = 1;
  const bool is_recording =
    ad_helper.start_recording_operations(tape_no /*material_id*/,
                                         true /*overwrite_tape*/,
                                         true /*keep*/);
  if (is_recording == true)
    {
      ad_helper.register_independent_variable(t, t_dof);
      ad_helper.register_independent_variable(s, s_dof);

      const Tensor<2, dim, ADNumberType> t_ad =
        ad_helper.get_sensitive_variables(t_dof);
      const ADNumberType s_ad = ad_helper.get_sensitive_variables(s_dof);

      const ADNumberType psi(func_ad::psi(t_ad, s_ad));

      ad_helper.register_dependent_variable(psi);
      ad_helper.stop_recording_operations(false /*write_tapes_to_file*/);


      std::cout << "Recorded data..." << std::endl;
      std::cout << "independent variable values: " << std::flush;
      ad_helper.print_values(std::cout);
      std::cout << "t_ad: " << t_ad << std::endl;
      std::cout << "s_ad: " << s_ad << std::endl;
      std::cout << "psi: " << psi << std::endl;
      std::cout << std::endl;
    }
  else
    {
      Assert(is_recording == true, ExcInternalError());
    }

  // Do some work :-)
  // Set a new evaluation point
  if (AD::ADNumberTraits<ADNumberType>::is_taped == true)
    {
      std::cout
        << "Using tape with different values for independent variables..."
        << std::endl;
      ad_helper.activate_recorded_tape(tape_no);
      s = 1.2;
      t *= 1.75;
      ad_helper.set_independent_variable(t, t_dof);
      ad_helper.set_independent_variable(s, s_dof);
    }

  std::cout << "independent variable values: " << std::flush;
  ad_helper.print_values(std::cout);

  // Compute the function value, gradient and hessian for the new evaluation
  // point
  psi = ad_helper.compute_value();
  ad_helper.compute_gradient(Dpsi);
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      ad_helper.compute_hessian(D2psi);
    }

  // Output the full stored function, gradient vector and hessian matrix
  std::cout << "psi: " << psi << std::endl;
  std::cout << "Dpsi: \n";
  Dpsi.print(std::cout);
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      std::cout << "D2psi: \n";
      D2psi.print_formatted(std::cout, 3, true, 0, "0.0");
    }

  // Extract components of the solution
  const Tensor<2, dim, ScalarNumberType> dpsi_dt =
    ad_helper.extract_gradient_component(Dpsi, t_dof);
  const ScalarNumberType dpsi_ds =
    ad_helper.extract_gradient_component(Dpsi, s_dof);
  std::cout << "extracted Dpsi (t): " << dpsi_dt << "\n"
            << "extracted Dpsi (s): " << dpsi_ds << "\n";

  // Verify the result
  using func = FunctionsTestTensorScalarCoupled<dim, ScalarNumberType>;
  static const ScalarNumberType tol =
    1e5 * std::numeric_limits<ScalarNumberType>::epsilon();

  Assert(std::abs(psi - func::psi(t, s)) < tol,
         ExcMessage("No match for function value."));
  Assert(std::abs((dpsi_dt - func::dpsi_dt(t, s)).norm()) < tol,
         ExcMessage("No match for first derivative."));
  Assert(std::abs(dpsi_ds - func::dpsi_ds(t, s)) < tol,
         ExcMessage("No match for first derivative."));
  if (AD::ADNumberTraits<ADNumberType>::n_supported_derivative_levels >= 2)
    {
      const Tensor<4, dim, ScalarNumberType> d2psi_dt_dt =
        ad_helper.extract_hessian_component(D2psi, t_dof, t_dof);
      const Tensor<2, dim, ScalarNumberType> d2psi_ds_dt =
        ad_helper.extract_hessian_component(D2psi, t_dof, s_dof);
      const Tensor<2, dim, ScalarNumberType> d2psi_dt_ds =
        ad_helper.extract_hessian_component(D2psi, t_dof, s_dof);
      const ScalarNumberType d2psi_ds_ds =
        ad_helper.extract_hessian_component(D2psi, s_dof, s_dof);
      std::cout << "extracted D2psi (t,t): " << d2psi_dt_dt << "\n"
                << "extracted D2psi (t,s): " << d2psi_ds_dt << "\n"
                << "extracted D2psi (s,t): " << d2psi_dt_ds << "\n"
                << "extracted D2psi (s,s): " << d2psi_ds_ds << "\n"
                << std::endl;
      Assert(std::abs((d2psi_dt_dt - func::d2psi_dt_dt(t, s)).norm()) < tol,
             ExcMessage("No match for second derivative."));
      Assert(std::abs((d2psi_ds_dt - func::d2psi_ds_dt(t, s)).norm()) < tol,
             ExcMessage("No match for second derivative."));
      Assert(std::abs((d2psi_dt_ds - func::d2psi_dt_ds(t, s)).norm()) < tol,
             ExcMessage("No match for second derivative."));
      Assert(std::abs(d2psi_ds_ds - func::d2psi_ds_ds(t, s)) < tol,
             ExcMessage("No match for second derivative."));
    }
}
