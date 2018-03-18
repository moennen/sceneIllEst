/*!
 * *****************************************************************************
 *   \file phSpline.h
 *   \author moennen
 *   \brief implementation of a polyharmonic spline interpolant
 *   \date 2018-03-16
 *   *****************************************************************************/
#ifndef _UTILS_PHSPLINE_H
#define _UTILS_PHSPLINE_H

#include <Eigen/Dense>
#include <vector>

template <unsigned InDim = 2, unsigned OutDim = InDim, unsigned Order = 2,
          typename Real = float>
class PhSpline final {
  const size_t _nCtrlPts;
  // Params of the spline : OutDim x ( _nCtrPts ( weights )  + InDim + 1  (
  // affine ) )
  std::vector<Real> _params;

  template <typename t, int dim = Eigen::Dynamic>
  using Vector = Eigen::Matrix<t, dim, 1>;
  using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

 public:
  PhSpline(const Real* ctrlPts, const Real* ctrlFunc, const size_t nCtrlPts)
      : _nCtrlPts(nCtrlPts), _params(OutDim * (_nCtrlPts + InDim + 1)) {
    // reject unsolvable system
    assert(_nCtrlPts > InDim);
    fit(ctrlPts, ctrlFunc, _nCtrlPts, &_params[0]);
  }

  // given a point x of InDim dimensions return the corresponding function value
  // y
  // in OutDim dimensions
  inline void operator()(const Real* ctrlPts, const Real* x, Real* y) const {
    using namespace Eigen;
    const size_t nParams = _nCtrlPts + InDim + 1;
    const auto mapX = Map<const Vector<Real, InDim> >(x);
    const auto mapCtrlPts = Map<const Matrix>(ctrlPts, InDim, _nCtrlPts);
    Vector<Real> vecX(nParams);
    for (unsigned pt = 0; pt < _nCtrlPts; ++pt) {
      vecX[pt] = psi((mapX - mapCtrlPts.col(pt)).norm());
    }
    vecX.template segment<InDim>(_nCtrlPts) = mapX;
    vecX[nParams - 1] = 1.0;

    // compute the interpolated function value
    auto vecY = Map<Vector<Real, OutDim> >(y);
    vecY = vecX.adjoint() * Map<const Matrix>(&_params[0], nParams, OutDim);
  }

  inline size_t getNParams() const { return _nCtrlPts + InDim + 1; }
  inline const Real* getParams() const & { return &_params[0]; }

 private:
  static inline Real psi(Real r) {
    return (Order % 2)
               ? std::pow(r, Order)
               : r < 1.0f ? std::pow(r, Order - 1) * std::log(std::pow(r, r))
                          : std::pow(r, Order) * std::log(r);
  }

  // estimate the weight st
  // _w = argmin_w ( ||eval(ref,w,ctrl) - probe||^2 )
  static inline void fit(const Real* ctrlPts, const Real* ctrlFuncs,
                         const size_t nCtrlPts, Real* params) {
    using namespace Eigen;
    // system matrix
    // -->
    const size_t nParams = nCtrlPts + InDim + 1;
    Matrix S(nParams, nParams);
    //#pragma omp parallel for
    for (size_t r = 0; r < nCtrlPts; ++r) {
      auto rx = Map<const Vector<Real, InDim> >(&ctrlPts[r * InDim]);
      for (unsigned c = 0; c < nCtrlPts; ++c) {
        auto cx = Map<const Vector<Real, InDim> >(&ctrlPts[c * InDim]);
        S(r, c) = psi((rx - cx).norm());
      }
    }
    //#pragma omp parallel for
    for (size_t c = 0; c < nCtrlPts; ++c) {
      auto cx = Map<const Vector<Real, InDim> >(&ctrlPts[c * InDim]);
      S.col(c).template segment<InDim>(nCtrlPts) = cx;
      S.row(c).template segment<InDim>(nCtrlPts) = cx;
    }
    S.template block<InDim + 1, InDim + 1>(nCtrlPts, nCtrlPts).setZero();
    S.col(nParams - 1).topRows(nCtrlPts).setOnes();
    S.row(nParams - 1).leftCols(nCtrlPts).setOnes();

    Matrix StS(nParams, nParams);
    StS.template triangularView<Lower>() = S.transpose() * S;

    Vector<Real> f(nParams);
    f.template bottomRows<InDim + 1>().setZero();
    for (size_t d = 0; d < OutDim; ++d) {
      for (size_t c = 0; c < nCtrlPts; ++c) {
        f[c] = ctrlFuncs[c * OutDim + d];
      }
      Matrix Stf = S.transpose() * f;
      StS.ldlt().solveInPlace(Stf);
      Map<Vector<Real> >(&params[d * nParams], nParams) = Stf;
    }
  }
};

#endif  // _UTILS_PHSPLINE_H
