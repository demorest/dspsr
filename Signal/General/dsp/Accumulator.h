//-*-C++-*-

#ifndef __Accumulator_h_
#define __Accumulator_h_

#include "environ.h"

#include "Transformation.h"
#include "TimeSeries.h"

namespace dsp {

  class Accumulator : public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    Accumulator();

    //! Virtual destructor
    virtual ~Accumulator();

    //! Reset the output
    void reset();

    //! Set the output max_samps (maximum ndat)
    void set_max_samps(uint64 _max_samps){ max_samps = _max_samps; }

    //! Inquire the output max_samps (maximum ndat)
    uint64 get_max_samps(){ return max_samps; }

  protected:

    //! Do the work
    void transformation();

  private:

    //! Set to true on reset
    bool append;

    //! The maximum ndat of the output
    uint64 max_samps;

  };

}

#endif

