//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/FilterbankConfig.h,v $
   $Revision: 1.1 $
   $Date: 2011/07/15 02:56:38 $
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
      After
    };

    Config ();

    void set_nchan (unsigned);
    unsigned get_nchan () const { return nchan; }

    void set_dedisperse_when (When);
    When get_dedisperse_when () const { return when; }

  protected:

    unsigned nchan;
    When when;

  };

  //! Insertion operator
  std::ostream& operator << (std::ostream&, const Filterbank::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, Filterbank::Config&);
}

#endif
