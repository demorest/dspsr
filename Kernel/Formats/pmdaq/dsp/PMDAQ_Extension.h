//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_PMDAQ_Extension_h_
#define __dsp_PMDAQ_Extension_h_

#include "dsp/Observation.h"
#include "dsp/dspExtension.h"

namespace dsp {

  class PMDAQ_Extension : public dspExtension {

  public:

    //! Null constructor
    PMDAQ_Extension();

    //! Copy constructor
    PMDAQ_Extension(const PMDAQ_Extension& p);

    //! Assignment operator
    PMDAQ_Extension& operator=(const PMDAQ_Extension& p);

    //! Virtual destructor
    virtual ~PMDAQ_Extension();

    //! Return a new copy-constructed instance identical to this instance
    virtual dspExtension* clone() const;

    void set_chan_begin(unsigned _chan_begin){ chan_begin = _chan_begin; }
    unsigned get_chan_begin() const { return chan_begin; }
    void set_chan_end(unsigned _chan_end){ chan_end = _chan_end; }
    unsigned get_chan_end() const { return chan_end; }

  private:

    //! First channel for the unpacker to start downloading [0]
    unsigned chan_begin;
    
    //! First channel for the unpacker to not download [99999]
    unsigned chan_end;

  };

}

#endif
