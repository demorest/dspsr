/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <string>

#include "environ.h"
#include "string_utils.h"

#include "dsp/PMDAQ_Extension.h"

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

//! Return a new null-constructed instance
dsp::dspExtension* dsp::PMDAQ_Extension::new_extension() const {
  return new PMDAQ_Extension;
}

//! Dump out to a string
string dsp::PMDAQ_Extension::dump_string() const{
  string s = "pmdaq_begin_chan\t" + make_string(chan_begin) + "\n";
  s += "pmdaq_end_chan\t" + make_string(chan_end) + "\n";
  return s;
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
