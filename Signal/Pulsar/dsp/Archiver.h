//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Archiver.h,v $
   $Revision: 1.2 $
   $Date: 2003/01/31 16:00:36 $
   $Author: wvanstra $ */


#ifndef __Archiver_h
#define __Archiver_h

#include "Pulsar/Archive.h"
#include "dsp/PhaseSeriesUnloader.h"

namespace dsp {

  //! Class to unload PhaseSeries data in a Pulsar::Archive
  /*! 

  */
  class Archiver : public PhaseSeriesUnloader {

  public:

    //! Verbose flag
    static bool verbose;

    //! Constructor
    Archiver () {  }
    
    //! Destructor
    virtual ~Archiver ();

    //! Unloads PhaseSeries data to a new Archive instance
    void unload (const PhaseSeries* data);

    void set_agent (Pulsar::Archive::Agent* agent);

    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

    void set (Pulsar::Integration* integration, const PhaseSeries* phase);

    void set (Pulsar::Profile* profile, const PhaseSeries* phase,
		      unsigned ichan, unsigned ipol, unsigned idim);


  protected:
    Reference::To<Pulsar::Archive::Agent> agent;

  };

}

#endif // !defined(__Archiver_h)
