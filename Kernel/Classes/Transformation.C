/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Transformation.h"
#include "dsp/SetBufferingPolicy.h"

Callback<dsp::TransformationBase*> dsp::TransformationBase::initialization;

//dsp::TransformationBase::initialization.connect (&SetBufferingPolicy::set);
