//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Archiver.h,v $
   $Revision: 1.3 $
   $Date: 2003/06/16 16:32:27 $
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

    //! Set the name of the Archive class to be used in creating new instances
    void set_archive_class (const char* archive_class_name);

    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

    void set (Pulsar::Integration* integration, const PhaseSeries* phase);

    void set (Pulsar::Profile* profile, const PhaseSeries* phase,
		      unsigned ichan, unsigned ipol, unsigned idim);


  protected:
    string archive_class_name;

  };

}

#endif // !defined(__Archiver_h)
