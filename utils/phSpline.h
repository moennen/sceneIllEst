/*! *****************************************************************************
 *   \file phSpline.h
 *   \author moennen
 *   \brief
 *   \date 2018-03-16
 *   *****************************************************************************/
#ifndef _UTILS_PHSPLINE_H
#define _UTILS_PHSPLINE_H

#include <Eigen/Dense>

template <unsigned NCtrls, unsigned InDim = 2, unsigned OutDim = InDim, unsigned Order = 2>
class PhSpline final
{
  private:
   // Ctrl points
   // [c1 c2 ... cN]
   Eigen::Matrix<float, InDim, NCtrls> _c;
   // Spline weights params
   // [w]
   Eigen::Matrix<float, NCtrls, OutDim> _w;
   // Spline affine params
   // [v]
   Eigen::Matrix<float, InDim + 1, OutDim> _v;

   // Vector used for evaluation
   mutable Eigen::Matrix<float, 1, NCtrls> _kv;

  public:
   using InPt = Eigen::Vector<float, InDim>;
   using OutPt = Eigen::Vector<float, OutDim>;

   PhSpline( const std::vector<InPt>& ctrlPts, const std::vector<OutPt>& ctrlFunc )
   {
      // set the ctrl points
      for ( size_t c=0; c<NCtrls; ++c ) _c.col(c) = ctrlPts[c];
      // set the ctrl values
      Eigen::Matrix<float, OutDim, NCtrls> f;
      for ( size_t c=0; c<NCtrls; ++c ) f.col(c) = ctrlFunc[c];
      
      // fit the parameters
      fit(_c, f, _w, _v);
   }

   inline OutPt operator()( const InPt& x ) const
   {
      // compute the kernel with every control points
      for ( unsigned c = 0; c < NCtrls; ++c ) _kv[c] = psi( ( x - _c.col( c ) ).norm() );
      // construct the homogenous vector
      const Eigen::Matrix<float, InDim + 1, 1> xh << x, Eigen::Matrix<float, 1, 1>::Ones();
      // compute the interpolated function value
      return ( _kv * _w + xh.transpose() * _v ).transpose();
   }

  private:
   static inline float psi( float r )
   {
      return ( Order % 2 ) ? std::pow( r, Order )
                           : r < 1.0f ? std::pow( r, Order - 1 ) * std::log( std::pow( r, r ) )
                                      : std::pow( r, Order ) * std::log( r );
   }

   // estimate the weight st
   // _w = argmin_w ( ||eval(ref,w,ctrl) - probe||^2 )
   static inline void fit( const Eigen::Matrix<float, InDim, NCtrls>& c,
                           const Eigen::Matrix<float, OutDim, NCtrls>& f,
                           Eigen::Matrix<float, NCtrls, OutDim>& w,
                           Eigen::Matrix<float, InDim + 1, OutDim> v )
   {
      // construct the rbf matrix
      Eigen::Matrix<float, NCtrls, NCtrls> A;
#pragma omp parallel for
      for ( size_t r = 0; r < NCtrls; ++r )
         for ( unsigned c = 0; c < NCtrls; ++c )
         {
            A( r, c ) = psi( _c.col( r ) - _c.col( r ) ).norm();
            // compute the RBF : norm^2 * log(norm)
             = psi( norm );
         }
      }

      // minimize ||A * x_i - b_i ||^2 with x_i = _w.row(i) and b_i = _probes.row(i) for every
      // output dimension
      //
      _w.resize( outDim, getNbCtrl() );
      Eigen::MatrixXd AtA( getNbRefs(), getNbCtrl() );
      AtA.triangularView<Eigen::Lower>() = A.transpose() * A;
      for ( size_t r = 0; r < outDim; ++r )
      {
         Eigen::MatrixXd Atb = A.transpose() * _probes.row( r ).transpose();
         AtA.ldlt().solveInPlace( Atb );
         _w.row( r ) = Atb.transpose();
      }
   }
};

#endif  // _UTILS_PHSPLINE_H
