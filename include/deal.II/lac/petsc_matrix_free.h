// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2012 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_petsc_matrix_free_h
#define dealii_petsc_matrix_free_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_PETSC
#  include <deal.II/lac/exceptions.h>
#  include <deal.II/lac/petsc_matrix_base.h>
#  include <deal.II/lac/petsc_vector.h>
DEAL_II_NAMESPACE_OPEN



namespace PETScWrappers
{
  /**
   * Implementation of a parallel matrix class based on PETSc
   * <tt>MatShell</tt> matrix-type. This base class implements only the
   * interface to the PETSc matrix object, while all the functionality is
   * contained in the matrix-vector multiplication which must be reimplemented
   * in derived classes.
   *
   * This interface is an addition to the dealii::MatrixFree class to realize
   * user-defined matrix-classes together with PETSc solvers and
   * functionalities. See also the documentation of dealii::MatrixFree class
   * and step-37 and step-48.
   *
   * Similar to other matrix classes in namespaces PETScWrappers and
   * PETScWrappers::MPI, the MatrixFree class provides the usual matrix-vector
   * multiplication <tt>vmult(VectorBase &dst, const VectorBase &src)</tt>
   * which is pure virtual and must be reimplemented in derived classes.
   * Besides the usual interface, this class has a matrix-vector
   * multiplication <tt>vmult(Vec &dst, const Vec &src)</tt> taking PETSc Vec
   * objects, which will be called by <tt>matrix_free_mult(Mat A, Vec src, Vec
   * dst)</tt> registered as matrix-vector multiplication of this PETSc matrix
   * object. The default implementation of the vmult function in the base
   * class wraps the given PETSc vectors with the PETScWrappers::VectorBase
   * class and then calls the usual vmult function with the usual interface.
   *
   * @ingroup PETScWrappers
   * @ingroup Matrix1
   */
  class MatrixFree : public MatrixBase
  {
  public:
    /**
     * Default constructor. Create an empty matrix object.
     */
    MatrixFree();

    /**
     * Create a matrix object of dimensions @p m times @p n with communication
     * happening over the provided @p communicator.
     *
     * For the meaning of the @p local_rows and @p local_columns parameters,
     * see the PETScWrappers::MPI::SparseMatrix class documentation.
     *
     * As other PETSc matrices, also the matrix-free object needs to have
     * a size and to perform matrix vector multiplications efficiently in
     * parallel also @p local_rows and @p local_columns. But in contrast to
     * PETSc::SparseMatrix classes a PETSc matrix-free object does not need
     * any estimation of non_zero entries and has no option
     * <tt>is_symmetric</tt>.
     */
    MatrixFree(const MPI_Comm     communicator,
               const unsigned int m,
               const unsigned int n,
               const unsigned int local_rows,
               const unsigned int local_columns);

    /**
     * Create a matrix object of dimensions @p m times @p n with communication
     * happening over the provided @p communicator.
     *
     * As other PETSc matrices, also the matrix-free object needs to have
     * a size and to perform matrix vector multiplications efficiently in
     * parallel also @p local_rows and @p local_columns. But in contrast to
     * PETSc::SparseMatrix classes a PETSc matrix-free object does not need
     * any estimation of non_zero entries and has no option
     * <tt>is_symmetric</tt>.
     */
    MatrixFree(const MPI_Comm                   communicator,
               const unsigned int               m,
               const unsigned int               n,
               const std::vector<unsigned int> &local_rows_per_process,
               const std::vector<unsigned int> &local_columns_per_process,
               const unsigned int               this_process);

    /**
     * Constructor for the serial case: Same function as
     * <tt>MatrixFree()</tt>, see above, with <tt>communicator =
     * MPI_COMM_WORLD</tt>.
     */
    MatrixFree(const unsigned int m,
               const unsigned int n,
               const unsigned int local_rows,
               const unsigned int local_columns);

    /**
     * Constructor for the serial case: Same function as
     * <tt>MatrixFree()</tt>, see above, with <tt>communicator =
     * MPI_COMM_WORLD</tt>.
     */
    MatrixFree(const unsigned int               m,
               const unsigned int               n,
               const std::vector<unsigned int> &local_rows_per_process,
               const std::vector<unsigned int> &local_columns_per_process,
               const unsigned int               this_process);

    /**
     * Throw away the present matrix and generate one that has the same
     * properties as if it were created by the constructor of this class with
     * the same argument list as the present function.
     */
    void
    reinit(const MPI_Comm     communicator,
           const unsigned int m,
           const unsigned int n,
           const unsigned int local_rows,
           const unsigned int local_columns);

    /**
     * Throw away the present matrix and generate one that has the same
     * properties as if it were created by the constructor of this class with
     * the same argument list as the present function.
     */
    void
    reinit(const MPI_Comm                   communicator,
           const unsigned int               m,
           const unsigned int               n,
           const std::vector<unsigned int> &local_rows_per_process,
           const std::vector<unsigned int> &local_columns_per_process,
           const unsigned int               this_process);

    /**
     * Call the @p reinit() function above with <tt>communicator =
     * MPI_COMM_WORLD</tt>.
     */
    void
    reinit(const unsigned int m,
           const unsigned int n,
           const unsigned int local_rows,
           const unsigned int local_columns);

    /**
     * Call the @p reinit() function above with <tt>communicator =
     * MPI_COMM_WORLD</tt>.
     */
    void
    reinit(const unsigned int               m,
           const unsigned int               n,
           const std::vector<unsigned int> &local_rows_per_process,
           const std::vector<unsigned int> &local_columns_per_process,
           const unsigned int               this_process);

    /**
     * Release all memory and return to a state just like after having called
     * the default constructor.
     */
    void
    clear();

    /**
     * Matrix-vector multiplication: let <i>dst = M*src</i> with <i>M</i>
     * being this matrix.
     *
     * Source and destination must not be the same vector.
     *
     * Note that if the current object represents a parallel distributed
     * matrix (of type PETScWrappers::MPI::SparseMatrix), then both vectors
     * have to be distributed vectors as well. Conversely, if the matrix is
     * not distributed, then neither of the vectors may be.
     */
    virtual void
    vmult(VectorBase &dst, const VectorBase &src) const = 0;

    /**
     * Matrix-vector multiplication: let <i>dst = M<sup>T</sup>*src</i> with
     * <i>M</i> being this matrix. This function does the same as @p vmult()
     * but takes the transposed matrix.
     *
     * Source and destination must not be the same vector.
     *
     * Note that if the current object represents a parallel distributed
     * matrix then both vectors have to be distributed vectors as well.
     * Conversely, if the matrix is not distributed, then neither of the
     * vectors may be.
     */
    virtual void
    Tvmult(VectorBase &dst, const VectorBase &src) const = 0;

    /**
     * Adding Matrix-vector multiplication. Add <i>M*src</i> on <i>dst</i>
     * with <i>M</i> being this matrix.
     *
     * Source and destination must not be the same vector.
     *
     * Note that if the current object represents a parallel distributed
     * matrix then both vectors have to be distributed vectors as well.
     * Conversely, if the matrix is not distributed, then neither of the
     * vectors may be.
     */
    virtual void
    vmult_add(VectorBase &dst, const VectorBase &src) const = 0;

    /**
     * Adding Matrix-vector multiplication. Add <i>M<sup>T</sup>*src</i> to
     * <i>dst</i> with <i>M</i> being this matrix. This function does the same
     * as @p vmult_add() but takes the transposed matrix.
     *
     * Source and destination must not be the same vector.
     *
     * Note that if the current object represents a parallel distributed
     * matrix then both vectors have to be distributed vectors as well.
     * Conversely, if the matrix is not distributed, then neither of the
     * vectors may be.
     */
    virtual void
    Tvmult_add(VectorBase &dst, const VectorBase &src) const = 0;

    /**
     * The matrix-vector multiplication called by @p matrix_free_mult(). This
     * function can be reimplemented in derived classes for efficiency. The
     * default implementation copies the given vectors into
     * PETScWrappers::*::Vector and calls <tt>vmult(VectorBase &dst, const
     * VectorBase &src)</tt> which is purely virtual and must be reimplemented
     * in derived classes.
     */
    virtual void
    vmult(Vec &dst, const Vec &src) const;

  private:
    /**
     * Callback-function registered as the matrix-vector multiplication of
     * this matrix-free object called by PETSc routines. This function must be
     * static and takes a PETSc matrix @p A, and vectors @p src and @p dst,
     * where <i>dst = A*src</i>
     *
     * Source and destination must not be the same vector.
     *
     * This function calls <tt>vmult(Vec &dst, const Vec &src)</tt> which
     * should be reimplemented in derived classes.
     */
    static int
    matrix_free_mult(Mat A, Vec src, Vec dst);

    /**
     * Do the actual work for the respective @p reinit() function and the
     * matching constructor, i.e. create a matrix object. Getting rid of the
     * previous matrix is left to the caller.
     */
    void
    do_reinit(const MPI_Comm     comm,
              const unsigned int m,
              const unsigned int n,
              const unsigned int local_rows,
              const unsigned int local_columns);
  };



} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE

#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PETSC

#endif
