//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_OutputArchive_h_
#define __dsp_OutputArchive_h_

#include "dsp/dspExtension.h"

namespace Pulsar
{
  class Archive;
}

namespace dsp {

  //! Creates a new instance of Pulsar::Archive to be used for output
  class OutputArchive : public dspExtension {
	
  public:

    //! Constructor passes name to dspExtension base class
    OutputArchive (const char* name) : dspExtension (name) { }

    virtual Pulsar::Archive* new_Archive () const = 0;

  };

}

#endif
