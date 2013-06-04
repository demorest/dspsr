//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFN.h,v $
   $Revision: 1.14 $
   $Date: 2011/09/09 02:38:14 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFN_h
#define __baseband_dsp_LoadToFN_h

#include "dsp/LoadToF1.h"
#include "dsp/MultiThread.h"

class ThreadContext;

namespace dsp {

  class UnloaderShare;

  //! Multiple LoadToF threads
  class LoadToFN : public MultiThread
  {

  public:

    //! Constructor
    LoadToFN (LoadToF::Config*);
    
    //! Set the number of thread to be used
    void set_nthread (unsigned);

    //! Set the configuration to be used in prepare and run
    void set_configuration (LoadToF::Config*);

    //! Setup sharing
    void share ();

    //! Finish by ensuring that all buffered outputs are flushed
    void finish ();

  protected:

    //! Configuration parameters
    /*! call to set_configuration may precede set_nthread */
    Reference::To<LoadToF::Config> configuration;

    //! PhaseSeriesUnloader sharing
    std::vector< Reference::To<UnloaderShare> > unloader;

    //! The creator of new LoadToF threads
    virtual LoadToF* new_thread ();

    LoadToF* at (unsigned index);

    template <class T>
    bool prepare_subint_archival ();


  };

}

#endif // !defined(__LoadToFN_h)

