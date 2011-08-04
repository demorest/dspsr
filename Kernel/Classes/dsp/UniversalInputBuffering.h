//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __UniversalInputBuffering_h
#define __UniversalInputBuffering_h

#include "dsp/InputBuffering.h"

namespace dsp {

  //! Buffers any TimeSeries used by Transformation
  template<typename Class, typename Method>
  class UniversalInputBuffering : public InputBuffering
  {
  public:
    
    //! Default constructor
    UniversalInputBuffering (Class* c, Method m)
      : InputBuffering (static_cast<HasInput<TimeSeries>*>(target)) { instance = c; method = m ; }

    //! Get the TimeSeries to be buffered
    const TimeSeries* get_input () { return (instance->*method)(); }

  protected:
    Class* instance;
    Method method;
  };

  template<typename Class, typename Method>
  UniversalInputBuffering<Class, Method>*
  new_UniversalInputBuffering (Class* target, Method method)
  {
    return new UniversalInputBuffering<Class, Method> (target, method);
  }

}

#endif // !defined(__InputBuffering_h)
