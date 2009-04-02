/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <string>
#include <stdio.h>

using namespace std;

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

