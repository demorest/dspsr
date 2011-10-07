//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FilterbankEngine_h
#define __FilterbankEngine_h

#include "dsp/Filterbank.h"
#include "dsp/filterbank_engine.h"

class dsp::Filterbank::Engine : public Reference::Able,
				public filterbank_engine
{
public:

  Engine () { scratch = output = 0; }

  //! If kernel is not set, then the engine should set up for benchmark only
  virtual void setup (Filterbank*) = 0;

  //! Perform the filterbank operation on the input data
  virtual void perform (const float* in) = 0;

  //! Finish up
  virtual void finish () { }

}; 

#endif
