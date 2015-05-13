//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_LoadToFITSN_h
#define __baseband_dsp_LoadToFITSN_h

#include "dsp/LoadToFITS.h"
#include "dsp/MultiThread.h"

class ThreadContext;

namespace dsp {

  class OutputFileShare;

  //! Multiple LoadToFITS threads
  class LoadToFITSN : public MultiThread
  {

  public:

    //! Constructor
    LoadToFITSN (LoadToFITS::Config*);
    
    //! Set the number of thread to be used
    void set_nthread (unsigned);

    //! Set the configuration to be used in prepare and run
    void set_configuration (LoadToFITS::Config*);

    //! Setup sharing
    void share ();

    //! Finish by ensuring that all buffered outputs are flushed
    void finish ();

  protected:

    //! Configuration parameters
    /*! call to set_configuration may precede set_nthread */
    Reference::To<LoadToFITS::Config> configuration;

    //! OutputFile sharing
    Reference::To<OutputFileShare> output_file;

    //! The creator of new LoadToFil threads
    virtual LoadToFITS* new_thread ();

    LoadToFITS* at (unsigned index);

  };

}

#endif // !defined(__LoadToFITSN_h)





