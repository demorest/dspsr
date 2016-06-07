//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_PolnSelect_h
#define __baseband_dsp_PolnSelect_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp
{
  //! Select only a single polarization from a TimeSeries
  class PolnSelect : public Transformation<TimeSeries,TimeSeries>
  {

  public:

    //! Default constructor
    PolnSelect ();

    //! Apply the npol to single-pol transformation
    void transformation ();

    //! Select the poln index to keep
    void set_ipol (int _ipol) { ipol_keep = _ipol; }

    // Get the currently selected poln index
    int get_ipol () const { return ipol_keep; }

    class Engine;

    void set_engine (Engine*);

  protected:

    //! The polarization to keep
    int ipol_keep;
  };

  class PolnSelect::Engine : public OwnStream
  {
  public:

    virtual void setup () = 0;

    virtual void fpt_polnselect (int ipol,
                                 const dsp::TimeSeries * in,
                                 dsp::TimeSeries * out) = 0;

    virtual void tfp_polnselect (int ipol,
                                 const dsp::TimeSeries* in,
                                 dsp::TimeSeries* out) = 0;

   };

}

#endif
