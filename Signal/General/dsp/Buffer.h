//-*-C++-*-

#ifndef __Buffer_h
#define __Buffer_h

#include <stdlib.h>

#include "Reference.h"

namespace dsp {
  
  template<class Container>
    class Buffer : public Reference::Able {
    public:
      enum BufferStatus{ free, full };

      Reference::To<Container> container;

      //! A class like RingBuffer doesn't actually need this, but I guess it's  nice to have is as an extra safety feature for debugging
      BufferStatus status;

      Buffer<Container>& operator=(const Buffer<Container>& buf){ container = buf.container; status = buf.status; return *this; }

      Buffer(const Buffer<Container>& buf){ operator=(buf); }
      Buffer(){ container = new Container; status = free; }

      ~Buffer(){ }
  };

}

#endif
