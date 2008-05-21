/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/dsp.h"

bool dsp::psrdisp_compatible = false;

/*!
  baseband/dsp version:

  1.0 - First version of the baseband/dsp library
  2.0 - dspsr at sourceforge, working with PSRCHIVE v 6.0

*/
const float dsp::version = 2.0;

Warning dsp::warn;
