//-*-C++-*-

#ifndef __SLDetect_h
#define __SLDetect_h

#include "Operation.h"

namespace dsp {

  //! Simply Square Law detects Timeseries in-place.

  class SLDetect : public Operation {

  public:

    SLDetect(Behaviour _type);
        
  protected:
    //! The operation loads the next block of data
    virtual void operation ();

  };

}

#endif // !defined(__SLDetect_h)
