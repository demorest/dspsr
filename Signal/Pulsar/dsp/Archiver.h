//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/Archiver.h,v $
   $Revision: 1.5 $
   $Date: 2003/06/16 22:03:54 $
   $Author: wvanstra $ */


#ifndef __Archiver_h
#define __Archiver_h

#include "Pulsar/Archive.h"
#include "dsp/PhaseSeriesUnloader.h"

namespace dsp {

  class Response;

  //! Class to unload PhaseSeries data in a Pulsar::Archive
  /*! 

  */
  class Archiver : public PhaseSeriesUnloader {

  public:

    //! Verbose flag
    static bool verbose;

    //! Constructor
    Archiver ();
    
    //! Destructor
    virtual ~Archiver ();

    //! Set the Response from which Passband Extension will be constructed
    void set_passband (const Response* passband);

    //! Unloads all available data to a new Archive instance
    void unload ();

    //! Set the name of the Pulsar::Archive class used to create new instances
    void set_archive_class (const char* archive_class_name);

    //! Set the Pulsar::Archive with the PhaseSeries data
    void set (Pulsar::Archive* archive, const PhaseSeries* phase);

  protected:

    //! Name of the Pulsar::Archive class used to create new instances
    string archive_class_name;

    //! Response from which Passband Extension will be constructed
    Reference::To<const Response> passband;

    //! Set the Response from which Passband Extension will be constructed
    void set_passband (Pulsar::Archive* archive);

    void set (Pulsar::Integration* integration, const PhaseSeries* phase);

    void set (Pulsar::Profile* profile, const PhaseSeries* phase,
		      unsigned ichan, unsigned ipol, unsigned idim);

  };

}

#endif // !defined(__Archiver_h)
