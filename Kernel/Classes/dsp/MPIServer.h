//-*-C++-*-

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
    vector< Reference::To<MPIRoot> > root;

  };

}

#endif // !defined(__SeekInput_h)
