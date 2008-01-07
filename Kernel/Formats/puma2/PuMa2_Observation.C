/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PuMa2_Observation.h"

#include "strutil.h"

using namespace std;

dsp::PuMa2_Observation::PuMa2_Observation (const char* header)
{
  if (header == NULL)
    throw Error (InvalidParam, "PuMa2_Observation", "no header!");

  load (header);

  set_mode (stringprintf ("%d-bit mode", get_nbit()));
  set_machine ("PuMa2");

}
