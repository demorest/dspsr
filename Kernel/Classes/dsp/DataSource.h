//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DataSource_h
#define __DataSource_h

#include "Reference.h"

namespace dsp {

  template<class Output>
  class DataSource : public Reference::Able {
  public:
    DataSource(){ }

    virtual ~DataSource(){ }
    
    virtual void propagate_go() = 0;
    virtual void propagate_stop() = 0;

    virtual bool get_full_buffer(Reference::To<Output>& buf) = 0;
    virtual void buffer_used() = 0;

    virtual void set_verbosity(bool _verbose) = 0;

  };

}

#endif
