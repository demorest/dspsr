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
dsp::BCPMExtension::BCPMExtension ()
  : dspExtension("BCPMExtension")
{
}

//! Copy constructor
dsp::BCPMExtension::BCPMExtension (const BCPMExtension& b) 
  : dspExtension("BCPMExtension")
{
  chtab = b.chtab;
}
