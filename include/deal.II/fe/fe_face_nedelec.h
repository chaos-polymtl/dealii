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

#ifndef dealii_fe_face_nedelec_h
#define dealii_fe_face_nedelec_h

#include <deal.II/base/config.h>

#include <deal.II/base/mutex.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_poly_tensor.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup fe
 * @{
 */

/**
 * A finite element that represents the <b>tangential trace</b> of an
 * H(curl)-conforming Nédélec field on the skeleton of the mesh faces. It is the
 * vector-valued, covariantly-transformed analogue of FE_TraceQ (which is the
 * trace of FE_Q): its degrees of freedom are exactly the edge and face degrees
 * of freedom of FE_Nedelec, while the cell-interior degrees of freedom are
 * dropped. The remaining degrees of freedom are tangentially continuous across
 * the edges shared by neighboring faces, so the element is H(curl)-conforming
 * on the face skeleton and is a natural trace/hybridization space for
 * H(curl) (Maxwell / eddy-current) problems.
 *
 * The polynomial space on each face is the corresponding lower-dimensional
 * Nédélec tangential space of the classic FE_Nedelec family:
 * - in 2d a "face" is an edge and the trace is the scalar tangential component
 *   of the field along that edge,
 * - in 3d a face is a quadrilateral and the trace is a two-component tangential
 *   field on that face.
 *
 * The constructor argument @p order is the Nédélec order, exactly as for
 * FE_Nedelec: the maximal polynomial degree of the shape functions is
 * `order + 1`, and FiniteElementData::degree is therefore one larger than the
 * value passed to the constructor.
 *
 * @note Because this element is defined through the tangential trace of
 * FE_Nedelec, its shape functions are constructed from the FE_Nedelec basis
 * functions associated with edges and faces. Unlike the scalar face elements
 * FE_FaceQ / FE_TraceQ, these basis functions are also non-zero in the interior
 * of a cell (they are a genuine subspace of FE_Nedelec). Evaluating this element
 * with an FEValues object therefore returns finite, well-defined values in the
 * cell interior; the element is nevertheless intended for use with FEFaceValues
 * and FESubfaceValues, where the tangential trace is the quantity of interest.
 *
 * @note Hanging-node constraints and the restriction and prolongation
 * matrices are implemented by reusing the corresponding structures of
 * FE_Nedelec: the interface constraints and the face and subface
 * interpolation matrices act only on face degrees of freedom, which this
 * element shares with FE_Nedelec, and the cell transfer matrices are the
 * leading principal blocks of FE_Nedelec's (the edge and face degrees of
 * freedom of the embedded or restricted function do not depend on the
 * interior ones). The element can therefore be used on adaptively refined
 * meshes. Non-standard face orientations are fully supported: the sign
 * changes of the edge degrees of freedom are inherited from FE_PolyTensor,
 * and the permutation and sign changes of the face degrees of freedom use
 * the same tables as FE_Nedelec.
 */
template <int dim>
class FE_FaceNedelec : public FE_PolyTensor<dim>
{
public:
  /**
   * Constructor for the tangential-trace Nédélec element of the given Nédélec
   * @p order (the maximal shape-function degree is `order + 1`).
   */
  FE_FaceNedelec(const unsigned int order);

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_FaceNedelec<dim>(order)</tt>, with @p dim and @p order
   * replaced by appropriate values.
   */
  virtual std::string
  get_name() const override;

  // documentation inherited from the base class
  virtual std::unique_ptr<FiniteElement<dim, dim>>
  clone() const override;

  /**
   * This function returns @p true, if the shape function @p shape_index has
   * non-zero function values somewhere on the face @p face_index.
   */
  virtual bool
  has_support_on_face(const unsigned int shape_index,
                      const unsigned int face_index) const override;

  // documentation inherited from the base class
  virtual void
  convert_generalized_support_point_values_to_dof_values(
    const std::vector<Vector<double>> &support_point_values,
    std::vector<double>               &nodal_values) const override;

  /**
   * Return the matrix interpolating from a face of one element to the face
   * of the neighboring element. Since the face degrees of freedom of this
   * element coincide with those of FE_Nedelec, the computation is delegated
   * to the underlying FE_Nedelec elements.
   *
   * Derived elements will have to implement this function. They may only
   * provide interpolation matrices for certain source finite elements, for
   * example those from the same family. If they don't implement
   * interpolation from a given element, then they must throw an exception of
   * type FiniteElement::ExcInterpolationNotImplemented.
   */
  virtual void
  get_face_interpolation_matrix(const FiniteElement<dim> &source,
                                FullMatrix<double> &interpolation_matrix,
                                const unsigned int  face_no = 0) const override;

  /**
   * Return the matrix interpolating from a face of one element to the
   * subface of the neighboring element. Since the face degrees of freedom of
   * this element coincide with those of FE_Nedelec, the computation is
   * delegated to the underlying FE_Nedelec elements.
   *
   * Derived elements will have to implement this function. They may only
   * provide interpolation matrices for certain source finite elements, for
   * example those from the same family. If they don't implement
   * interpolation from a given element, then they must throw an exception of
   * type FiniteElement::ExcInterpolationNotImplemented.
   */
  virtual void
  get_subface_interpolation_matrix(
    const FiniteElement<dim> &source,
    const unsigned int        subface,
    FullMatrix<double>       &interpolation_matrix,
    const unsigned int        face_no = 0) const override;

  /**
   * Return the prolongation (embedding) matrix of the given child for the
   * given refinement case. The matrix is the leading principal block of
   * FE_Nedelec's prolongation matrix: the edge and face degrees of freedom
   * of the embedded function do not depend on the interior degrees of
   * freedom of the coarse cell. Like in FE_Nedelec, the matrix is computed
   * (through FE_Nedelec) on first request.
   */
  virtual const FullMatrix<double> &
  get_prolongation_matrix(
    const unsigned int         child,
    const RefinementCase<dim> &refinement_case =
      RefinementCase<dim>::isotropic_refinement) const override;

  /**
   * Return the restriction matrix of the given child for the given
   * refinement case. The matrix is the leading principal block of
   * FE_Nedelec's restriction matrix: the coarse edge and face node
   * functionals act on the tangential trace on the coarse edges and faces,
   * which lie on the boundaries of the child cells and are therefore
   * determined by the child edge and face degrees of freedom alone. Like in
   * FE_Nedelec, the matrix is computed (through FE_Nedelec) on first
   * request.
   */
  virtual const FullMatrix<double> &
  get_restriction_matrix(
    const unsigned int         child,
    const RefinementCase<dim> &refinement_case =
      RefinementCase<dim>::isotropic_refinement) const override;

  // documentation inherited from the base class
  virtual bool
  hp_constraints_are_implemented() const override;

  // documentation inherited from the base class
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_vertex_dof_identities(const FiniteElement<dim> &fe_other) const override;

  // documentation inherited from the base class
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_line_dof_identities(const FiniteElement<dim> &fe_other) const override;

  // documentation inherited from the base class
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_quad_dof_identities(const FiniteElement<dim> &fe_other,
                         const unsigned int        face_no = 0) const override;

  /**
   * @copydoc FiniteElement::compare_for_domination()
   */
  virtual FiniteElementDomination::Domination
  compare_for_domination(const FiniteElement<dim> &fe_other,
                         const unsigned int codim = 0) const override final;

  // documentation inherited from the base class
  virtual std::pair<Table<2, bool>, std::vector<unsigned int>>
  get_constant_modes() const override;

private:
  /**
   * Only for internal use. Returns the @p dofs_per_object vector, i.e. the
   * degree-of-freedom distribution of FE_Nedelec with the cell-interior
   * degrees of freedom removed.
   */
  static std::vector<unsigned int>
  get_dpo_vector(const unsigned int order);

  /**
   * Initialize the permutation pattern and the pattern of sign change of the
   * face (quad) degrees of freedom for non-standard face orientations in 3d.
   * The face degrees of freedom of this element coincide with those of
   * FE_Nedelec, so this uses the same swap tables.
   *
   * @note Currently this is implemented for orders k < 5, as for FE_Nedelec.
   */
  void
  initialize_quad_dof_index_permutation_and_sign_change();

  /**
   * The underlying FE_Nedelec element of which this element represents the
   * tangential trace. It provides the generalized support points, the
   * interpolation of function values to degree-of-freedom values, the
   * per-face support information, the interface constraints, the face and
   * subface interpolation matrices, and the restriction and prolongation
   * matrices, all restricted to the edge and face degrees of freedom.
   */
  FE_Nedelec<dim> fe_nedelec;

  /**
   * Mutex variables used for protecting the on-demand computation of the
   * restriction and prolongation matrices.
   */
  mutable Threads::Mutex restriction_matrix_mutex;
  mutable Threads::Mutex prolongation_matrix_mutex;
};

/** @} */

DEAL_II_NAMESPACE_CLOSE

#endif
