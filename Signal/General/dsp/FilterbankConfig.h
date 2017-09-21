//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankConfig.h

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

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Set the stream information for the device
    void set_stream (void*);

    //! Return a new Filterbank instance and configure it
    Filterbank* create ();

  protected:

    Memory* memory;
    void* stream;
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
