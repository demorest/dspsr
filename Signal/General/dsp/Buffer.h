//-*-C++-*-

#ifndef __Buffer_h
#define __Buffer_h

#include <string>

#include <stdlib.h>

#include "environ.h"
#include "Reference.h"

namespace dsp {
  
  template<class Container>
    class Buffer : public Reference::Able {
    public:
      enum BufferStatus{ free, full };

      Reference::To<Container> container;

      //! A class like RingBuffer doesn't actually need this, but I guess it's  nice to have is as an extra safety feature for debugging
      BufferStatus status;

      //! The file the Buffer was loaded from
      std::string filename;

      //! The offset from the start of the file that the Buffer starts from
      uint64 offset;

      Buffer<Container>& operator=(const Buffer<Container>& buf);

      Buffer(const Buffer<Container>& buf);
      Buffer();

      ~Buffer();
  };

}

template<class Container>
dsp::Buffer<Container>::Buffer(){
  status = free;
  offset = 0;
  filename = "unset";

  container = new Container;
}

template<class Container>
dsp::Buffer<Container>::Buffer(const Buffer<Container>& buf){
  operator=(buf);
}

template<class Container>
dsp::Buffer<Container>::~Buffer(){ }

template<class Container>
dsp::Buffer<Container>& dsp::Buffer<Container>::operator=(const dsp::Buffer<Container>& buf){
  status = buf.status;
  offset = buf.offset;
  filename = buf.filename;

  container = buf.container;
  
  return *this;
}

#endif

