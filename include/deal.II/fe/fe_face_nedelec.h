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
 * the edges shared by neighbouring faces, so the element is H(curl)-conforming
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
 * @note The current implementation follows the deliberately lean approach of
 * FE_NedelecNodal: hp-hanging-node constraints, the non-standard face
 * orientation permutation of face degrees of freedom, and the
 * restriction/prolongation (multigrid) matrices are not yet provided. Standard
 * orientation and edge sign changes (inherited from FE_PolyTensor) are handled.
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
   * interpolation of function values to degree-of-freedom values, and the
   * per-face support information, all restricted to the edge and face degrees
   * of freedom.
   */
  FE_Nedelec<dim> fe_nedelec;
};

/** @} */

DEAL_II_NAMESPACE_CLOSE

#endif
