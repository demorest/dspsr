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

    //! Set the output subsize
    void set_subsize(uint64 _subsize){ subsize = _subsize; }

    //! Inquire the output subsize
    uint64 get_subsize(){ return subsize; }

  protected:

    //! Do the work
    void transformation();

  private:

    //! Set to true on reset
    bool append;

    //! The subsize of the output
    uint64 subsize;

  };

}

#endif

