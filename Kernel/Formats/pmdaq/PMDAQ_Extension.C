/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PMDAQ_Extension.h"
#include "tostring.h"

using namespace std;

//! Null constructor
dsp::PMDAQ_Extension::PMDAQ_Extension() : dspExtension("PMDAQ_Extension")
{
  chan_begin = 0;
  chan_end = 99999;
}

//! Virtual destructor
dsp::PMDAQ_Extension::~PMDAQ_Extension(){ }

//! Return a new copy-constructed instance identical to this instance
dsp::dspExtension* dsp::PMDAQ_Extension::clone() const{
  return new PMDAQ_Extension(*this);
}

//! Copy constructor
dsp::PMDAQ_Extension::PMDAQ_Extension(const PMDAQ_Extension& p) : dspExtension("PMDAQ_Extension"){
  operator=(p);
}

//! Assignment operator
dsp::PMDAQ_Extension& dsp::PMDAQ_Extension::operator=(const PMDAQ_Extension& p){
  if( &p == this )
    return *this;

  chan_begin = p.chan_begin;
  chan_end = p.chan_end;

  return *this;
}
