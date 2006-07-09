//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __MPIServer_h
#define __MPIServer_h

#include "Reference.h"

namespace dsp {

  class MPIRoot;

  //! Serves data from a number of MPIRoot instances
  class MPIServer : Reference::Able {
    
  public:
    
    //! Default constructor
    MPIServer ();

    //! Destructor
    virtual ~MPIServer ();

    //! Manage the MPIRoot instance
    void manage (MPIRoot* root);

    //! Serve the data from the managed MPIRoot instances
    void serve ();

  protected:

    //! The managed MPIRoot instances
    std::vector< Reference::To<MPIRoot> > root;

  };

}

#endif // !defined(__SeekInput_h)
