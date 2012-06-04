//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankConfig.h,v $
   $Revision: 1.2 $
   $Date: 2011/07/15 04:18:11 $
   $Author: straten $ */

#ifndef __FilterbankConfig_h
#define __FilterbankConfig_h

#include "dsp/Filterbank.h"

namespace dsp
{
  class Filterbank::Config
  {
  public:

    //! When dedispersion takes place with respect to filterbank
    enum When
    {
      Before,
      During,
      After,
      Never
    };

    Config ();

    void set_nchan (unsigned n) { nchan = n; }
    unsigned get_nchan () const { return nchan; }

    void set_freq_res (unsigned n) { freq_res = n; }
    unsigned get_freq_res () const { return freq_res; }

    void set_convolve_when (When w) { when = w; }
    When get_convolve_when () const { return when; }

  protected:

    unsigned nchan;
    unsigned freq_res;
    When when;

  };

  //! Insertion operator
  std::ostream& operator << (std::ostream&, const Filterbank::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, Filterbank::Config&);
}

#endif
