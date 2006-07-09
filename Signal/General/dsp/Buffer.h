//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __Buffer_h
#define __Buffer_h

#include <string>

#include <stdlib.h>

#include "environ.h"
#include "Reference.h"

#include "dsp/BasicBuffer.h"

namespace dsp {
  
  template<class Container>
    class Buffer : public BasicBuffer {
    public:
      Buffer(const Buffer<Container>& buf);
      Buffer();

      virtual ~Buffer();

      Buffer<Container>& operator=(const Buffer<Container>& buf);

      virtual Container* get_container(){ return container; }
      virtual void set_container(Container* _container)
      { container = _container; }
      
      virtual void vset_container(void* _container);
      virtual void* vget_container();

    protected:

      Reference::To<Container> container;
  };

}

template<class Container>
dsp::Buffer<Container>::Buffer() : BasicBuffer() {
  container = new Container;
}

template<class Container>
dsp::Buffer<Container>::Buffer(const Buffer<Container>& buf){
  operator=(buf);
}

template<class Container>
dsp::Buffer<Container>::~Buffer(){ }

template<class Container>
dsp::Buffer<Container>& dsp::Buffer<Container>::operator=(const Buffer<Container>& buf){
  BasicBuffer::operator=( buf );

  container = buf.container;
  
  return *this;
}

template<class Container>
void dsp::Buffer<Container>::vset_container(void* _container){
  set_container( (Container*)_container );
}

template<class Container>
void* dsp::Buffer<Container>::vget_container(){
  return (void*)get_container();
}

#endif

