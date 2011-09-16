//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_BCPMExtension_h_
#define __dsp_BCPMExtension_h_

#include "dsp/dspExtension.h"

namespace dsp{

  class BCPMExtension : public dspExtension {

  public:

    //! Null constructor
    BCPMExtension();

    //! Copy constructor
    BCPMExtension (const BCPMExtension&);

    //! Return a new copy-constructed instance identical to this instance
    BCPMExtension* clone() const { return new BCPMExtension(*this); }

    //! This stores the ordering of the channels in a BCPM data file
    std::vector<int> chtab;

  };

}

#endif
