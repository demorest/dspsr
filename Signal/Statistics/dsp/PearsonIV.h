//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PearsonIV_h
#define __PearsonIV_h

#include "dsp/Romberg.h"
#include "ReferenceAble.h"

namespace dsp {

  class PearsonIV : public Reference::Able {

  public:

    PearsonIV ( unsigned _M );
    ~PearsonIV ();

    //! Return the probability of SK value x
    double operator() (double x);

    //! Return cumulative function for x
    double cf (double x);
    double log_cf (double x);
    double dlog_cf (double x);

    //! Return complementary cumulative function for x
    double ccf (double x);
    double log_ccf (double x);
    double dlog_ccf (double x);

    unsigned get_M () { return SK_M; }
    double get_m () { return m; }
    double get_v () { return v; }
    double get_lamda () { return lamda; }
    double get_a () { return a; }
      
    typedef double argument_type;
    typedef double return_type;

  private:

    //! Calculate the first four moments of the distribution and ancilliary parameters
    void prepare();

    //double normalisation();
    double log_normal();

    //! Number of integrations in SK Statisic
    unsigned SK_M;

    //! first moment
    double mu1;
    
    //! second moment
    double mu2;

    //! third moment
    double beta1;

    //! fourth moment
    double beta2;

    double r;

    double m;

    double v;

    double a;

    double lamda;

    // pearson type IV normalisation factor
    //double k;
    double logk;

    Romberg<MidPoint> tricky;

    Romberg<> normal;

    unsigned verbose;
  };
}

#endif
