//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Archiver.h,v $
   $Revision: 1.1 $
   $Date: 2002/10/12 00:55:24 $
   $Author: wvanstra $ */


#ifndef __Archiver_h
#define __Archiver_h

#include "Reference.h"

// forward declaration
namespace Pulsar {
  class Archive;
  class Integration;
  class Profile;
}

namespace dsp {

  class PhaseSeries;

  //! Class to set the data in a Pulsar::Archive
  /*! 

  */
  class Archiver : public Reference::Able {

  public:

    //! Verbose flag
    static bool verbose;

    //! Constructor
    Archiver () {  }
    
    //! Destructor
    virtual ~Archiver () {  }
 
    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

    void set (Pulsar::Integration* integration, const PhaseSeries* phase);

    void set (Pulsar::Profile* profile, const PhaseSeries* phase,
		      unsigned ichan, unsigned ipol, unsigned idim);


  };

}

#endif // !defined(__Archiver_h)
