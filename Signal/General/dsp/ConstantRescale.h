//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_ConstantRescale_h
#define __baseband_dsp_ConstantRescale_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <vector>

class ThreadContext;

namespace dsp
{
  //! Rescale all channels and polarizations
  /*! This is a multi-thread aware version of rescale.  It will 
   * apply a constant scale/offset, with levels computed from the 
   * first available data block.  The same scale/offset will be shared
   * between all threads.
   */
  class ConstantRescale : public Transformation<TimeSeries,TimeSeries>
  {

    class ScaleOffsetShare;

  public:

    //! Default constructor
    ConstantRescale ();

    //! Desctructor
    ~ConstantRescale ();

    //! Rescale to zero mean and unit variance
    void transformation ();

  private:

    //! Compute the mean and variance of the current data
    void compute_levels();

    //! The scale/offset sharing class
    Reference::To<ScaleOffsetShare> share;

    std::vector< std::vector<float> > scale;
    std::vector< std::vector<float> > offset;
  };

  //! Share scale/offset between threads
  class ConstantRescale::ScaleOffsetShare : public Reference::Able
  {
  public:

    ScaleOffsetShare ();

    ~ScaleOffsetShare();

    //! Return the current scale/offset
    void get_scale_offset (const TimeSeries* input,
        std::vector< std::vector<float > >& scale, 
        std::vector< std::vector<float > >& offset);

    //! Compute scale/offset from the given data block
    void compute(const TimeSeries* input);

  private:

    //! Have levels been computed
    bool computed;

    //! Thread lock
    ThreadContext *context;

    std::vector< std::vector<float> > scale;
    std::vector< std::vector<float> > offset;

  };

}
#endif
