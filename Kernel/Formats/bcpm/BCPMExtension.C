#include "dsp/BCPMExtension.h"

//! Null constructor
dsp::BCPMExtension::BCPMExtension() : dspExtension("BCPMExtension"){ }

//! Virtual destructor
dsp::BCPMExtension::~BCPMExtension(){ }

//! Copy stuff
void dsp::BCPMExtension::copy(const dsp::BCPMExtension& b){
  if( &b==this )
    return;

  chtab = b.chtab;
}
