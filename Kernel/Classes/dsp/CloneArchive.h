//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_CloneArchive_h_
#define __dsp_CloneArchive_h_

#include "dsp/OutputArchive.h"

namespace dsp {

  //! Creates a cloned instance of Pulsar::Archive to be used for output
  class CloneArchive : public OutputArchive
  {
	
  public:

    //! Construct from Pulsar::Archive instance to be cloned
    CloneArchive (const Pulsar::Archive*);

    //! Copy constructor
    CloneArchive (const CloneArchive&);

    //! Destructor
    ~CloneArchive ();

    //! Clone operator
    dspExtension* clone () const;

    //! Return a clone of the Pulsar::Archive instance
    Pulsar::Archive* new_Archive () const;

  protected:

    Reference::To<const Pulsar::Archive> instance;

  };

}

#endif
