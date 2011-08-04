//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ZapWeightCUDA.h"

#include <memory>

using namespace std;

CUDA::ZapWeightEngine::ZapWeightEngine (cudaStream_t _stream)
{
  stream = _stream;
}

CUDA::ZapWeightEngine::~ZapWeightEngine ()
{
}

void CUDA::ZapWeightEngine::setup ()
{

  if (dsp::Operation::verbose)
    cerr << "CUDA::ZapWeightEngine::setup()" << endl;
}

void CUDA::ZapWeightEngine::perform ()
{

  if (dsp::Operation::verbose)
    cerr << "CUDA::ZapWeightEngine::perform()" << endl;
}

