/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Transformation.h"
#include "dsp/SetBufferingPolicy.h"

using namespace dsp;

Callback<TransformationBase*> TransformationBase::initialization;

static int setup_initialization ()
{
  TransformationBase::initialization.connect (SetBufferingPolicy::set);
  return 0;
}

static int setup = setup_initialization ();

