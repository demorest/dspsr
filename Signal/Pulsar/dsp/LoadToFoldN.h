//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFoldN.h,v $
   $Revision: 1.2 $
   $Date: 2007/05/31 05:59:40 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFoldN_h
#define __baseband_dsp_LoadToFoldN_h

#include "dsp/LoadToFold1.h"

class ThreadContext;

namespace dsp {

  //! Multiple LoadToFold threads
  class LoadToFoldN : public LoadToFold {

  public:

    //! Constructor
    LoadToFoldN (unsigned nthread = 0);
    
    //! Destructor
    ~LoadToFoldN ();

    //! Set the number of thread to be used
    void set_nthread (unsigned);

    //! Set the configuration to be used in prepare and run
    void set_configuration (Config*);

    //! Set the Input from which data will be read
    void set_input (Input*);

    //! Prepare to fold the input TimeSeries
    void prepare ();

    //! Run through the data
    void run ();

    //! Finish everything
    void finish ();

  protected:

    //! Configuration parameters
    /*! call to set_configuration may precede set_nthread */
    Reference::To<Config> configuration;

    //! Input
    /*! call to set_input may precede set_nthread */
    Reference::To<Input> input;

    //! Thread lock for Input::load
    ThreadContext* input_context;

    //! The creator of new LoadToFold1 threads
    virtual LoadToFold1* new_thread ();

    //! The LoadToFold1 threads
    std::vector< Reference::To<LoadToFold1> > threads;

    //! The thread ids
    std::vector<pthread_t> ids;

  };

}

#endif // !defined(__LoadToFoldN_h)





