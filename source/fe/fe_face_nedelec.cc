// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/polynomials_nedelec.h>
#include <deal.II/base/tensor_polynomials_base.h>

#include <deal.II/fe/fe_face_nedelec.h>
#include <deal.II/fe/fe_nothing.h>

#include <algorithm>
#include <memory>
#include <sstream>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  namespace FE_FaceNedelecImplementation
  {
    // Number of FE_Nedelec basis functions associated with the interior
    // (cell) degrees of freedom for a given Nédélec order. These are exactly
    // the ones that the tangential-trace element FE_FaceNedelec drops.
    template <int dim>
    unsigned int
    n_interior_polynomials(const unsigned int order)
    {
      switch (dim)
        {
          case 2:
            // interior (quad) dofs of FE_Nedelec<2>
            return 2 * order * (order + 1);
          case 3:
            // interior (hex) dofs of FE_Nedelec<3>
            return 3 * order * order * (order + 1);
          default:
            DEAL_II_NOT_IMPLEMENTED();
            return 0;
        }
    }



    // Number of FE_Nedelec basis functions that live on the boundary
    // (i.e. on edges and, in 3d, faces). Because FE_Nedelec numbers its degrees
    // of freedom (and, since its inverse node matrix is the identity, its
    // polynomial basis) as edge dofs, then face dofs, then interior dofs, these
    // are exactly the first n_boundary_polynomials() basis functions.
    template <int dim>
    unsigned int
    n_boundary_polynomials(const unsigned int order)
    {
      return PolynomialsNedelec<dim>::n_polynomials(order) -
             n_interior_polynomials<dim>(order);
    }



    /**
     * The tangential-trace Nédélec polynomial space. It exposes the first
     * n_boundary_polynomials() basis functions of PolynomialsNedelec, i.e. the
     * ones associated with the edge and face degrees of freedom, which form the
     * H(curl) tangential trace on the face skeleton.
     */
    template <int dim>
    class PolynomialsFaceNedelec : public TensorPolynomialsBase<dim>
    {
    public:
      PolynomialsFaceNedelec(const unsigned int order)
        : TensorPolynomialsBase<dim>(order, n_boundary_polynomials<dim>(order))
        , full_space(order)
      {}

      void
      evaluate(const Point<dim>            &unit_point,
               std::vector<Tensor<1, dim>> &values,
               std::vector<Tensor<2, dim>> &grads,
               std::vector<Tensor<3, dim>> &grad_grads,
               std::vector<Tensor<4, dim>> &third_derivatives,
               std::vector<Tensor<5, dim>> &fourth_derivatives) const override
      {
        const unsigned int n_boundary = this->n();
        const unsigned int n_full     = full_space.n();

        Assert(values.empty() || values.size() == n_boundary,
               ExcDimensionMismatch(values.size(), n_boundary));
        Assert(grads.empty() || grads.size() == n_boundary,
               ExcDimensionMismatch(grads.size(), n_boundary));
        Assert(grad_grads.empty() || grad_grads.size() == n_boundary,
               ExcDimensionMismatch(grad_grads.size(), n_boundary));
        Assert(third_derivatives.empty() ||
                 third_derivatives.size() == n_boundary,
               ExcDimensionMismatch(third_derivatives.size(), n_boundary));
        Assert(fourth_derivatives.empty() ||
                 fourth_derivatives.size() == n_boundary,
               ExcDimensionMismatch(fourth_derivatives.size(), n_boundary));

        // Evaluate the full Nédélec space into temporary vectors (sized to the
        // full space where the caller asked for a quantity, and empty where it
        // did not) and copy the leading, boundary-associated entries.
        std::vector<Tensor<1, dim>> full_values(values.empty() ? 0 : n_full);
        std::vector<Tensor<2, dim>> full_grads(grads.empty() ? 0 : n_full);
        std::vector<Tensor<3, dim>> full_grad_grads(grad_grads.empty() ? 0 :
                                                                          n_full);
        std::vector<Tensor<4, dim>> full_third(
          third_derivatives.empty() ? 0 : n_full);
        std::vector<Tensor<5, dim>> full_fourth(
          fourth_derivatives.empty() ? 0 : n_full);

        full_space.evaluate(unit_point,
                            full_values,
                            full_grads,
                            full_grad_grads,
                            full_third,
                            full_fourth);

        for (unsigned int i = 0; i < n_boundary; ++i)
          {
            if (!values.empty())
              values[i] = full_values[i];
            if (!grads.empty())
              grads[i] = full_grads[i];
            if (!grad_grads.empty())
              grad_grads[i] = full_grad_grads[i];
            if (!third_derivatives.empty())
              third_derivatives[i] = full_third[i];
            if (!fourth_derivatives.empty())
              fourth_derivatives[i] = full_fourth[i];
          }
      }

      std::string
      name() const override
      {
        return "FaceNedelec";
      }

      std::unique_ptr<TensorPolynomialsBase<dim>>
      clone() const override
      {
        return std::make_unique<PolynomialsFaceNedelec<dim>>(*this);
      }

    private:
      PolynomialsNedelec<dim> full_space;
    };

  } // namespace FE_FaceNedelecImplementation
} // namespace internal



template <int dim>
FE_FaceNedelec<dim>::FE_FaceNedelec(const unsigned int order)
  : FE_PolyTensor<dim>(
      internal::FE_FaceNedelecImplementation::PolynomialsFaceNedelec<dim>(order),
      FiniteElementData<dim>(get_dpo_vector(order),
                             dim,
                             order + 1,
                             FiniteElementData<dim>::Hcurl),
      std::vector<bool>(
        internal::FE_FaceNedelecImplementation::n_boundary_polynomials<dim>(
          order),
        true),
      std::vector<ComponentMask>(
        internal::FE_FaceNedelecImplementation::n_boundary_polynomials<dim>(
          order),
        ComponentMask(std::vector<bool>(dim, true))))
  , fe_nedelec(order)
{
  Assert(dim >= 2, ExcImpossibleInDim(dim));

  this->mapping_kind = {mapping_nedelec};

  // We reuse the FE_Nedelec basis directly for the edge and face degrees of
  // freedom, so - just as for FE_Nedelec - no basis transformation from the
  // polynomial space to the node functionals is required.
  this->inverse_node_matrix.clear();

  // The generalized support points and the interpolation of function values to
  // degree-of-freedom values are delegated to the underlying FE_Nedelec (see
  // convert_generalized_support_point_values_to_dof_values() below), so we use
  // the same generalized support points here. Note that this set also contains
  // the cell-interior support points of FE_Nedelec; they do not influence the
  // edge/face degrees of freedom of this element and are simply carried along.
  this->generalized_support_points = fe_nedelec.get_generalized_support_points();

  // The interface constraints (constraining the degrees of freedom on a
  // refined face to those on the unrefined neighboring face) of FE_Nedelec
  // are built from face embedding matrices and thus involve only face (edge
  // and quad) degrees of freedom, which this element shares with FE_Nedelec.
  // They can therefore be copied verbatim.
  this->interface_constraints = fe_nedelec.constraints();

  // Finally set up the permutation and sign changes of the face (quad) degrees
  // of freedom for non-standard face orientations in 3d. The edge (line)
  // degrees of freedom only need sign changes, which are handled at fill time
  // by internal::FE_PolyTensor::get_dof_sign_change_nedelec().
  initialize_quad_dof_index_permutation_and_sign_change();
}



template <int dim>
std::string
FE_FaceNedelec<dim>::get_name() const
{
  // note that the FETools::get_fe_by_name function depends on the particular
  // format of the string this function returns, so they have to be kept in sync

  // this->degree is the maximal polynomial degree and is thus one higher than
  // the Nédélec order given to the constructor.
  std::ostringstream namebuf;
  namebuf << "FE_FaceNedelec<" << dim << ">(" << this->degree - 1 << ")";
  return namebuf.str();
}



template <int dim>
std::unique_ptr<FiniteElement<dim, dim>>
FE_FaceNedelec<dim>::clone() const
{
  return std::make_unique<FE_FaceNedelec<dim>>(this->degree - 1);
}



template <int dim>
bool
FE_FaceNedelec<dim>::has_support_on_face(const unsigned int shape_index,
                                         const unsigned int face_index) const
{
  AssertIndexRange(shape_index, this->n_dofs_per_cell());
  // The edge and face degrees of freedom of this element share the numbering of
  // FE_Nedelec (they are its leading degrees of freedom), so we can simply ask
  // the underlying FE_Nedelec for the face-support information.
  return fe_nedelec.has_support_on_face(shape_index, face_index);
}



template <int dim>
void
FE_FaceNedelec<dim>::convert_generalized_support_point_values_to_dof_values(
  const std::vector<Vector<double>> &support_point_values,
  std::vector<double>               &nodal_values) const
{
  AssertDimension(support_point_values.size(),
                  this->generalized_support_points.size());
  AssertDimension(nodal_values.size(), this->n_dofs_per_cell());

  // Delegate to the underlying FE_Nedelec, which computes all edge, face and
  // interior degree-of-freedom values from the same set of generalized support
  // point values. The edge and face degrees of freedom are the leading ones and
  // are computed independently of the interior ones, so we simply keep those.
  std::vector<double> full_nodal_values(fe_nedelec.n_dofs_per_cell());
  fe_nedelec.convert_generalized_support_point_values_to_dof_values(
    support_point_values, full_nodal_values);

  std::copy(full_nodal_values.begin(),
            full_nodal_values.begin() + this->n_dofs_per_cell(),
            nodal_values.begin());
}



template <int dim>
void
FE_FaceNedelec<dim>::get_face_interpolation_matrix(
  const FiniteElement<dim> &source,
  FullMatrix<double>       &interpolation_matrix,
  const unsigned int        face_no) const
{
  Assert(interpolation_matrix.m() == source.n_dofs_per_face(face_no),
         ExcDimensionMismatch(interpolation_matrix.m(),
                              source.n_dofs_per_face(face_no)));
  Assert(interpolation_matrix.n() == this->n_dofs_per_face(face_no),
         ExcDimensionMismatch(interpolation_matrix.n(),
                              this->n_dofs_per_face(face_no)));

  // The face degrees of freedom of this element coincide with those of
  // FE_Nedelec, and so do the face interpolation matrices. Delegate to the
  // underlying FE_Nedelec elements.
  if (const FE_FaceNedelec<dim> *source_fe =
        dynamic_cast<const FE_FaceNedelec<dim> *>(&source))
    {
      fe_nedelec.get_face_interpolation_matrix(source_fe->fe_nedelec,
                                               interpolation_matrix,
                                               face_no);
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&source) != nullptr)
    {
      // nothing to do here, the FE_Nothing has no degrees of freedom anyway
    }
  else
    AssertThrow(
      false, (typename FiniteElement<dim>::ExcInterpolationNotImplemented()));
}



template <int dim>
void
FE_FaceNedelec<dim>::get_subface_interpolation_matrix(
  const FiniteElement<dim> &source,
  const unsigned int        subface,
  FullMatrix<double>       &interpolation_matrix,
  const unsigned int        face_no) const
{
  Assert(interpolation_matrix.m() == source.n_dofs_per_face(face_no),
         ExcDimensionMismatch(interpolation_matrix.m(),
                              source.n_dofs_per_face(face_no)));
  Assert(interpolation_matrix.n() == this->n_dofs_per_face(face_no),
         ExcDimensionMismatch(interpolation_matrix.n(),
                              this->n_dofs_per_face(face_no)));

  // As in get_face_interpolation_matrix(), delegate to the underlying
  // FE_Nedelec elements.
  if (const FE_FaceNedelec<dim> *source_fe =
        dynamic_cast<const FE_FaceNedelec<dim> *>(&source))
    {
      fe_nedelec.get_subface_interpolation_matrix(source_fe->fe_nedelec,
                                                  subface,
                                                  interpolation_matrix,
                                                  face_no);
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&source) != nullptr)
    {
      // nothing to do here, the FE_Nothing has no degrees of freedom anyway
    }
  else
    AssertThrow(
      false, (typename FiniteElement<dim>::ExcInterpolationNotImplemented()));
}



template <int dim>
const FullMatrix<double> &
FE_FaceNedelec<dim>::get_prolongation_matrix(
  const unsigned int         child,
  const RefinementCase<dim> &refinement_case) const
{
  AssertIndexRange(refinement_case,
                   RefinementCase<dim>::isotropic_refinement + 1);
  Assert(refinement_case != RefinementCase<dim>::no_refinement,
         ExcMessage(
           "Prolongation matrices are only available for refined cells!"));
  AssertIndexRange(child, this->reference_cell().n_children(refinement_case));

  std::lock_guard<std::mutex> lock(prolongation_matrix_mutex);

  // initialization upon first request
  if (this->prolongation[refinement_case - 1][child].n() == 0)
    {
      // The prolongation (embedding) of FE_Nedelec is exact, and the interior
      // shape functions have vanishing tangential trace on the cell boundary.
      // The edge and face degrees of freedom of the embedded function on a
      // child cell therefore depend only on the edge and face degrees of
      // freedom on the coarse cell, i.e., the prolongation matrix of this
      // element is the leading principal block of FE_Nedelec's.
      const FullMatrix<double> &full_matrix =
        fe_nedelec.get_prolongation_matrix(child, refinement_case);

      // need to get a non-const reference in order to be able to fill the
      // matrix inside a const function
      FullMatrix<double> &this_matrix =
        const_cast<FE_FaceNedelec<dim> &>(*this)
          .prolongation[refinement_case - 1][child];
      this_matrix.reinit(this->n_dofs_per_cell(), this->n_dofs_per_cell());
      for (unsigned int i = 0; i < this->n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < this->n_dofs_per_cell(); ++j)
          this_matrix(i, j) = full_matrix(i, j);
    }

  return this->prolongation[refinement_case - 1][child];
}



template <int dim>
const FullMatrix<double> &
FE_FaceNedelec<dim>::get_restriction_matrix(
  const unsigned int         child,
  const RefinementCase<dim> &refinement_case) const
{
  AssertIndexRange(refinement_case,
                   RefinementCase<dim>::isotropic_refinement + 1);
  Assert(refinement_case != RefinementCase<dim>::no_refinement,
         ExcMessage(
           "Restriction matrices are only available for refined cells!"));
  AssertIndexRange(child, this->reference_cell().n_children(refinement_case));

  std::lock_guard<std::mutex> lock(restriction_matrix_mutex);

  // initialization upon first request
  if (this->restriction[refinement_case - 1][child].n() == 0)
    {
      // The coarse edge and face node functionals act on the tangential
      // trace on the coarse edges and faces. These lie on the boundaries of
      // the child cells, where the tangential trace is determined by the
      // child edge and face degrees of freedom alone, i.e., the restriction
      // matrix of this element is the leading principal block of
      // FE_Nedelec's.
      const FullMatrix<double> &full_matrix =
        fe_nedelec.get_restriction_matrix(child, refinement_case);

      // need to get a non-const reference in order to be able to fill the
      // matrix inside a const function
      FullMatrix<double> &this_matrix =
        const_cast<FE_FaceNedelec<dim> &>(*this)
          .restriction[refinement_case - 1][child];
      this_matrix.reinit(this->n_dofs_per_cell(), this->n_dofs_per_cell());
      for (unsigned int i = 0; i < this->n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < this->n_dofs_per_cell(); ++j)
          this_matrix(i, j) = full_matrix(i, j);
    }

  return this->restriction[refinement_case - 1][child];
}



template <int dim>
bool
FE_FaceNedelec<dim>::hp_constraints_are_implemented() const
{
  return true;
}



template <int dim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_FaceNedelec<dim>::hp_vertex_dof_identities(const FiniteElement<dim> &) const
{
  // This element has no degrees of freedom on vertices.
  return std::vector<std::pair<unsigned int, unsigned int>>();
}



template <int dim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_FaceNedelec<dim>::hp_line_dof_identities(
  const FiniteElement<dim> &fe_other) const
{
  // Two line (edge) degrees of freedom are identical if their edge shape
  // functions have the same polynomial degree.
  if (const FE_FaceNedelec<dim> *fe_face_nedelec_other =
        dynamic_cast<const FE_FaceNedelec<dim> *>(&fe_other))
    {
      std::vector<std::pair<unsigned int, unsigned int>> identities;
      const unsigned int p_min =
        std::min(fe_face_nedelec_other->degree, this->degree);
      identities.reserve(p_min);
      for (unsigned int i = 0; i < p_min; ++i)
        identities.emplace_back(i, i);
      return identities;
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
    {
      return std::vector<std::pair<unsigned int, unsigned int>>();
    }
  else
    {
      DEAL_II_NOT_IMPLEMENTED();
      return std::vector<std::pair<unsigned int, unsigned int>>();
    }
}



template <int dim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_FaceNedelec<dim>::hp_quad_dof_identities(const FiniteElement<dim> &fe_other,
                                            const unsigned int) const
{
  // Two face (quad) degrees of freedom are identical if their face shape
  // functions have the same polynomial degree. This mirrors FE_Nedelec, whose
  // face degrees of freedom are shared with this element.
  if (const FE_FaceNedelec<dim> *fe_face_nedelec_other =
        dynamic_cast<const FE_FaceNedelec<dim> *>(&fe_other))
    {
      const unsigned int p     = fe_face_nedelec_other->degree;
      const unsigned int q     = this->degree;
      const unsigned int p_min = std::min(p, q);
      std::vector<std::pair<unsigned int, unsigned int>> identities;

      for (unsigned int i = 0; i < p_min; ++i)
        for (unsigned int j = 0; j < p_min - 1; ++j)
          {
            identities.emplace_back(i * (q - 1) + j, i * (p - 1) + j);
            identities.emplace_back(i + (j + q - 1) * q, i + (j + p - 1) * p);
          }

      return identities;
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
    {
      return std::vector<std::pair<unsigned int, unsigned int>>();
    }
  else
    {
      DEAL_II_NOT_IMPLEMENTED();
      return std::vector<std::pair<unsigned int, unsigned int>>();
    }
}



template <int dim>
FiniteElementDomination::Domination
FE_FaceNedelec<dim>::compare_for_domination(const FiniteElement<dim> &fe_other,
                                            const unsigned int codim) const
{
  Assert(codim <= dim, ExcImpossibleInDim(dim));
  (void)codim;

  if (const FE_FaceNedelec<dim> *fe_face_nedelec_other =
        dynamic_cast<const FE_FaceNedelec<dim> *>(&fe_other))
    {
      if (this->degree < fe_face_nedelec_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_face_nedelec_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_Nothing<dim> *fe_nothing =
             dynamic_cast<const FE_Nothing<dim> *>(&fe_other))
    {
      if (fe_nothing->is_dominating())
        return FiniteElementDomination::other_element_dominates;
      else
        return FiniteElementDomination::no_requirements;
    }

  DEAL_II_NOT_IMPLEMENTED();
  return FiniteElementDomination::neither_element_dominates;
}



template <int dim>
std::pair<Table<2, bool>, std::vector<unsigned int>>
FE_FaceNedelec<dim>::get_constant_modes() const
{
  Table<2, bool> constant_modes(dim, this->n_dofs_per_cell());
  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int i = 0; i < this->n_dofs_per_cell(); ++i)
      constant_modes(d, i) = true;

  std::vector<unsigned int> components;
  components.reserve(dim);
  for (unsigned int d = 0; d < dim; ++d)
    components.push_back(d);

  return std::pair<Table<2, bool>, std::vector<unsigned int>>(constant_modes,
                                                              components);
}



template <int dim>
void
FE_FaceNedelec<dim>::initialize_quad_dof_index_permutation_and_sign_change()
{
  // The face (quad) degrees of freedom of this element coincide with those of
  // FE_Nedelec: there are 2*k*(k+1) of them per quad, indexed as
  //
  // | x0, x1, ..., xk | y0, y1, ..., yk |
  // |-- half_dofs=k*(k+1) --|-- half_dofs=k*(k+1) --|
  //
  // For non-standard face orientations in 3d these have to be permuted and may
  // change sign. The swap tables below are taken verbatim from FE_Nedelec (see
  // FE_Nedelec::initialize_quad_dof_index_permutation_and_sign_change() for a
  // detailed description of their format); they encode, for each of the eight
  // combined face orientations, the y-dof each x-dof is swapped with together
  // with the sign changes of the x- and y-dofs.

  static const int c_swap_table_0 = 0;

  static const int c_swap_table_1[8][3][2] = {           // 0   1
                                              {{-1, -1}, // 0
                                               {0, 0},
                                               {0, 0}},
                                              {{0, 1}, // 1
                                               {0, 0},
                                               {0, 0}},
                                              {{0, 1}, // 2
                                               {1, 0},
                                               {0, 0}},
                                              {{-1, -1}, // 3
                                               {0, 0},
                                               {1, 0}},
                                              {{-1, -1}, // 4
                                               {1, 0},
                                               {1, 0}},
                                              {{0, 1}, // 5
                                               {1, 0},
                                               {1, 0}},
                                              {{0, 1}, // 6
                                               {0, 0},
                                               {1, 0}},
                                              {{-1, -1}, // 7
                                               {1, 0},
                                               {0, 0}}};

  static const int c_swap_table_2[8][3][6] = {// 0   1   2   3   4   5
                                              {{-1, -1, -1, -1, -1, -1}, // 0
                                               {0, 0, 0, 0, 0, 0},
                                               {0, 0, 0, 0, 0, 0}},
                                              {{0, 3, 1, 4, 2, 5}, // 1
                                               {0, 0, 0, 0, 0, 0},
                                               {0, 0, 0, 0, 0, 0}},
                                              {{0, 3, 1, 4, 2, 5}, // 2
                                               {1, 1, 0, 0, 1, 1},
                                               {0, 0, 0, 1, 1, 1}},
                                              {{-1, -1, -1, -1, -1, -1}, // 3
                                               {0, 1, 0, 1, 0, 1},
                                               {1, 0, 1, 1, 0, 1}},
                                              {{-1, -1, -1, -1, -1, -1}, // 4
                                               {1, 0, 0, 1, 1, 0},
                                               {1, 0, 1, 0, 1, 0}},
                                              {{0, 3, 1, 4, 2, 5}, // 5
                                               {1, 0, 0, 1, 1, 0},
                                               {1, 0, 1, 0, 1, 0}},
                                              {{0, 3, 1, 4, 2, 5}, // 6
                                               {0, 1, 0, 1, 0, 1},
                                               {1, 0, 1, 1, 0, 1}},
                                              {{-1, -1, -1, -1, -1, -1}, // 7
                                               {1, 1, 0, 0, 1, 1},
                                               {0, 0, 0, 1, 1, 1}}};

  static const int c_swap_table_3[8][3][12] = {
    // 0   1   2   3   4   5   6   7   8   9  10  11
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 0
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}, // 1
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}, // 2
     {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0},
     {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}},
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 3
     {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0},
     {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}},
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 4
     {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
     {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0}},
    {{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}, // 5
     {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
     {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0}},
    {{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}, // 6
     {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0},
     {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}},
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 7
     {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0},
     {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}}};

  static const int c_swap_table_4[8][3][20] = {
    // 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
    // 19
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 0
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{0,  5,  10, 15, 1,  6,  11, 16, 2,  7,
      12, 17, 3,  8,  13, 18, 4,  9,  14, 19}, // 1
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{0,  5,  10, 15, 1,  6,  11, 16, 2,  7,
      12, 17, 3,  8,  13, 18, 4,  9,  14, 19}, // 2
     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1}},
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 3
     {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1},
     {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 4
     {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
     {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0}},
    {{0,  5,  10, 15, 1,  6,  11, 16, 2,  7,
      12, 17, 3,  8,  13, 18, 4,  9,  14, 19}, // 5
     {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
     {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0}},
    {{0,  5,  10, 15, 1,  6,  11, 16, 2,  7,
      12, 17, 3,  8,  13, 18, 4,  9,  14, 19}, // 6
     {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1},
     {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // 7
     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1}}};

  static const int *swap_table_array[5] = {&c_swap_table_0,
                                           &c_swap_table_1[0][0][0],
                                           &c_swap_table_2[0][0][0],
                                           &c_swap_table_3[0][0][0],
                                           &c_swap_table_4[0][0][0]};

  static const int row_length[5] = {0, 2, 6, 12, 20};
  static const int table_size[5] = {
    0, 8 * 3 * 2, 8 * 3 * 6, 8 * 3 * 12, 8 * 3 * 20};

  // Only three-dimensional elements have shared quad dofs that need permuting;
  // in 2d the quad is the cell interior (no shared dofs) and the line dofs are
  // sign-adjusted at fill time.
  if (dim != 3)
    return;

  const unsigned int k = this->tensor_degree() - 1;

  // The lowest-order element has no quad dofs.
  if (k == 0)
    return;

  // Orders > 4 are not implemented.
  AssertThrow(k < 5, ExcNotImplemented());

  // The implementation assumes that all quads have the same number of dofs.
  AssertDimension(this->n_unique_faces(), 1);
  const unsigned int face_no = 0;

  Assert(
    this->adjust_quad_dof_index_for_face_orientation_table[0].n_elements() ==
      this->reference_cell().n_face_orientations(face_no) *
        this->n_dofs_per_quad(face_no),
    ExcInternalError());

  Assert(
    this->adjust_quad_dof_sign_for_face_orientation_table[0].n_elements() ==
      this->reference_cell().n_face_orientations(face_no) *
        this->n_dofs_per_quad(face_no),
    ExcInternalError());

  Assert(2 * k * (k + 1) == this->n_dofs_per_quad(face_no), ExcInternalError());

  const int *swap_table = swap_table_array[k];

  const unsigned int half_dofs = k * (k + 1);

  const int rl = row_length[k];
  for (types::geometric_orientation combined_orientation = 0;
       combined_orientation <
       this->reference_cell().n_face_orientations(face_no);
       ++combined_orientation)
    {
      for (unsigned int index_x = 0; index_x < half_dofs; index_x++)
        {
          int offset = 3 * rl * combined_orientation + 0 * rl + index_x;
          Assert(offset < table_size[k], ExcInternalError());
          int value = *(swap_table + offset);
          Assert(value < table_size[k], ExcInternalError());
          Assert(value > -2, ExcInternalError());

          if (value != -1)
            {
              const unsigned int index_y =
                half_dofs + static_cast<unsigned int>(value);

              // dofs swap
              this->adjust_quad_dof_index_for_face_orientation_table[face_no](
                index_x, combined_orientation) = index_y - index_x;

              this->adjust_quad_dof_index_for_face_orientation_table[face_no](
                index_y, combined_orientation) = index_x - index_y;
            }

          // dof sign change
          offset = 3 * rl * combined_orientation + 1 * rl + index_x;
          Assert(offset < table_size[k], ExcInternalError());
          value = *(swap_table + offset);
          Assert((value == 0) || (value == 1), ExcInternalError());

          this->adjust_quad_dof_sign_for_face_orientation_table[face_no](
            index_x, combined_orientation) = static_cast<bool>(value);


          offset = 3 * rl * combined_orientation + 2 * rl + index_x;
          Assert(offset < table_size[k], ExcInternalError());
          value = *(swap_table + offset);
          Assert((value == 0) || (value == 1), ExcInternalError());

          this->adjust_quad_dof_sign_for_face_orientation_table[face_no](
            index_x + half_dofs, combined_orientation) =
            static_cast<bool>(value);
        }
    }

  return;
}



template <int dim>
std::vector<unsigned int>
FE_FaceNedelec<dim>::get_dpo_vector(const unsigned int order)
{
  // The same distribution as FE_Nedelec, but with the interior (cell) degrees
  // of freedom removed: only edge and (in 3d) face degrees of freedom remain.
  std::vector<unsigned int> dpo(dim + 1, 0U);
  dpo[1] = order + 1; // edge (line) dofs
  if (dim == 3)
    dpo[2] = 2 * order * (order + 1); // face (quad) dofs
  // dpo[dim] (interior) stays zero

  return dpo;
}



// explicit instantiations
#include "fe/fe_face_nedelec.inst"

DEAL_II_NAMESPACE_CLOSE
