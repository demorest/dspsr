/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <string>
#include <stdio.h>

#include "psr_cpp.h"

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

//! Dump out to a string
string dsp::BCPMExtension::dump_string() const {
  string ss = "BCPMExtension MiniExtension.C $Revision: 1.4 $ $Date: 2006/07/09 13:27:03 $:\n";

  char dummy[4096];

  for( unsigned i=0; i<chtab.size(); i++){
    sprintf(dummy,"%d %d\n",i,chtab[i]);
    ss += dummy;
  }

  return ss;
}
