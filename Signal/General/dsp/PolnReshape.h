//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_PolnReshape_h
#define __baseband_dsp_PolnReshape_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

// Detection is handled very efficiently on the GPU for 2pol, analytic data.
// But other formats may be more useful down the signal path.  This
// Transformation allows post-detection conversion from 2pol,2dim data to
// a variety of formats:

// In the 2pol, 2dim case the coherency parameters are packed such that
// AA and BB are the 2 dimensions of a sample in the pol=0 stream, and
// AB and BA* are the 2 dimensions of a sample in the pol=1 stream.

// npol=4, ndim = 1: Coherence / Stokes
// npol=2, ndim = 1: PPQQ
// npol=1, ndim = 1: Intensity

// The transformation performed is determined uniquely by the output state.

namespace dsp
{
  //! Convert npol=2,ndim=2 format to variety of other data shapes
  class PolnReshape : public Transformation<TimeSeries,TimeSeries>
  {

  public:

    //! Default constructor
    PolnReshape ();

    //! Apply the npol to single-pol transformation
    void transformation ();

    //! Reshape the poln index to keep
    void set_state ( Signal::State _state) { state = _state; }

  protected:

    //! The polarization to keep
    Signal::State state;

    //! Handle 2x2 --> 4x1 (Coherence or Stokes)
    void p2d2_p4d1();

    //! Handle 2x2 --> 2x1 (PPQQ)
    void p2d2_p2d1();

    //! Handle 2x2 --> 1x1 (Intensity)
    void p2d2_p1d1();

    //! Handle 1x1 --> 1x1 (Intensity)
    void p1d1_p1d1();

  };
}

#endif
