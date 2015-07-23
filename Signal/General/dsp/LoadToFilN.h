//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2014 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_dsp_LoadToFilN_h
#define __baseband_dsp_LoadToFilN_h

#include "dsp/LoadToFil.h"
#include "dsp/MultiThread.h"

class ThreadContext;

namespace dsp {

  class OutputFileShare;

  //! Multiple LoadToFil threads
  class LoadToFilN : public MultiThread
  {

  public:

    //! Constructor
    LoadToFilN (LoadToFil::Config*);
    
    //! Set the number of thread to be used
    void set_nthread (unsigned);

    //! Set the configuration to be used in prepare and run
    void set_configuration (LoadToFil::Config*);

    //! Setup sharing
    void share ();

    //! Finish by ensuring that all buffered outputs are flushed
    void finish ();

  protected:

    //! Configuration parameters
    /*! call to set_configuration may precede set_nthread */
    Reference::To<LoadToFil::Config> configuration;

    //! OutputFile sharing
    Reference::To<OutputFileShare> output_file;

    //! The creator of new LoadToFil threads
    virtual LoadToFil* new_thread ();

    LoadToFil* at (unsigned index);

    // XXX not sure if needed
    //template <class T>
    //bool prepare_subint_archival ();


  };

}

#endif // !defined(__LoadToFilN_h)





