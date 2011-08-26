//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/LoadToFoldN.h,v $
   $Revision: 1.13 $
   $Date: 2011/08/26 22:05:06 $
   $Author: straten $ */

#ifndef __baseband_dsp_LoadToFoldN_h
#define __baseband_dsp_LoadToFoldN_h

#include "dsp/LoadToFold1.h"
#include "dsp/MultiThread.h"

class ThreadContext;

namespace dsp {

  class UnloaderShare;

  //! Multiple LoadToFold threads
  class LoadToFoldN : public MultiThread
  {

  public:

    //! Constructor
    LoadToFoldN (LoadToFold::Config*);
    
    //! Set the number of thread to be used
    void set_nthread (unsigned);

    //! Set the configuration to be used in prepare and run
    void set_configuration (LoadToFold::Config*);

    //! Setup sharing
    void share ();

  protected:

    //! Configuration parameters
    /*! call to set_configuration may precede set_nthread */
    Reference::To<LoadToFold::Config> configuration;

    //! PhaseSeriesUnloader sharing
    std::vector< Reference::To<UnloaderShare> > unloader;

    //! The creator of new LoadToFold threads
    virtual LoadToFold* new_thread ();

    LoadToFold* at (unsigned index);

    template <class T>
    bool prepare_subint_archival ();


  };

}

#endif // !defined(__LoadToFoldN_h)





