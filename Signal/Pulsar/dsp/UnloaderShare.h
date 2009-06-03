//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/UnloaderShare.h,v $
   $Revision: 1.10 $
   $Date: 2009/06/03 07:41:22 $
   $Author: straten $ */

#ifndef __UnloaderShare_h
#define __UnloaderShare_h

#include "dsp/PhaseSeriesUnloader.h"
#include "dsp/SignalPath.h"
#include "dsp/TimeDivide.h"

class ThreadContext;

namespace dsp {

  class SubFold;
  class PhaseSeries;
  class PhaseSeriesUnloader;

  //! Unload PhaseSeries data from multiple threads
  /*! 
  The SubFold class from each thread is given a pointer to an instance
  of the nested Submit class, which acts as a controlled interface to
  the UnloaderShare that created it.
  */
  class UnloaderShare : public Reference::Able {

  public:

    //! Constructor
    UnloaderShare (unsigned contributors);
    
    //! Destructor
    virtual ~UnloaderShare ();

    //! Set the context for control of shared resources
    void set_context (ThreadContext*);

    //! When sub-integration is finished, wait for all other threads to finish
    void set_wait_all (bool);

    //! The PhaseSeries submission interface
    class Submit;

    //! Get submission interface for the specified contributor
    Submit* new_Submit (unsigned contributor);

    //! Unload the PhaseSeries data
    void unload (const PhaseSeries*, unsigned contributor);

    //! Inform any waiting threads that contributor is finished
    void finish_all (unsigned contributor);

    //! Copy the Divider configuration
    void copy (const TimeDivide*);

    //! Set the start time from which to begin counting sub-integrations
    void set_start_time (const MJD& start_time);

    //! Get the start time from which to begin counting sub-integrations
    MJD get_start_time () const;

    //! Set the number of seconds to fold into each sub-integration
    void set_subint_seconds (double subint_seconds);

    //! Get the number of seconds to fold into each sub-integration
    double get_subint_seconds () const;

    //! Set the number of turns to fold into each sub-integration
    void set_subint_turns (unsigned subint_turns);

    //! Get the number of turns to fold into each sub-integration
    unsigned get_subint_turns () const;

    /** @name deprecated methods 
     *  Use of these methods is deprecated in favour of attaching
     *  callback methods to the completed event. */
    //@{

    //! Set the shared file unloader
    void set_unloader (PhaseSeriesUnloader* unloader);

    //! Get the shared file unloader
    PhaseSeriesUnloader* get_unloader () const;

    //! Unload all cached subintegrations
    void finish ();

  protected:

    class Storage;

    //! Unload the storage
    void unload (Storage*);

    //! Temporary storage of incomplete sub-integrations
    std::vector< Reference::To<Storage> > storage;

    //! File unloader
    Reference::To<PhaseSeriesUnloader> unloader;
    
    //! The time divider
    TimeDivide divider;

    //! The number of contributors
    unsigned contributors;

    //! The last division completed by a contributor
    std::vector<uint64> last_division;

    //! Thread coordination used in unload method
    ThreadContext* context;

    //! First contributor to complete a division waits for all others
    bool wait_all;

    //! Flags set when a contributor calls finish_all
    std::vector<bool> finished_all;

  };

  class UnloaderShare::Submit : public PhaseSeriesUnloader
  {
  public:

    //! Default constructor
    Submit (UnloaderShare* parent, unsigned contributor);

    //! Unload the PhaseSeries data
    void unload (const PhaseSeries*);

    //! Inform any waiting threads that the current thread is finished
    void finish ();

    //! Set the minimum integration length required to unload data
    void set_minimum_integration_length (double seconds);

  protected:

    Reference::To<UnloaderShare> parent;
    unsigned contributor;
  };

  class UnloaderShare::Storage : public Reference::Able
  {
  public:

    //! Default constructor
    Storage (unsigned contributors, const std::vector<bool>& finished);

    //! ~Destructor
    ~Storage ();

    //! Set the storage area
    void set_profiles( PhaseSeries* );

    //! Get the storage area
    PhaseSeries* get_profiles ();

    //! Set the division
    void set_division( uint64 );

    //! Get the division
    uint64 get_division ();

    //! Register the last division finished by the specified contributor
    bool integrate (unsigned contributor, uint64 division, const PhaseSeries*);

    //! Inform any waiting threads that contributor is finished this division
    void set_finished (unsigned contributor);

    //! Return true when all contributors are finished with this integration
    bool get_finished ();

    //! Wait for all threads to complete
    void wait_all (ThreadContext*);

  protected:

    Reference::To<PhaseSeries> profiles;

    std::vector<bool> finished;
    uint64 division;

    //! Thread coordination used in integrate and wait_all
    ThreadContext* context;
    void print_finished ();

  };

}

#endif // !defined(__UnloaderShare_h)


