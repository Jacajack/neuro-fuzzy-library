/** @file */ 

#ifndef S_NORM_SUM_H
#define S_NORM_SUM_H

#include <iostream>
#include "s-norm.h"

namespace ksi
{
   struct s_norm_sum : public s_norm
   {
      /** value of S-norm: \f$ S(a, b) = a + b - ab \f$ */
      virtual double snorm (double, double) const;
      virtual s_norm * clone() const ;
      
      /** The method prints an object into output stream.
      * @param ss an output stream to print to
      */
      virtual std::ostream & Print (std::ostream & ss) const; 
      
      virtual ~s_norm_sum(); 
   };
}

#endif
