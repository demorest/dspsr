//-*-C++-*-

#ifndef __dsp_BCPMExtension_h_
#define __dsp_BCPMExtension_h_

#include <vector>

#include "dsp/dspExtension.h"

namespace dsp{

  class BCPMExtension : public dspExtension {

  public:

    //! Null constructor
    BCPMExtension();

    //! Copy constructor
    BCPMExtension(const BCPMExtension& b) : dspExtension("BCPMExtension") { copy(b); }

    //! Virtual destructor
    virtual ~BCPMExtension();

    //! Return a new copy-constructed instance identical to this instance
    BCPMExtension* clone() const { return new BCPMExtension(*this); }

    //! Return a new null-constructed instance
    BCPMExtension* new_extension() const { return new BCPMExtension; }

    //! Copy stuff
    virtual void copy(const BCPMExtension& b);
    
    //! This stores the ordering of the channels in a BCPM data file
    vector<int> chtab;

  };

}

#endif
