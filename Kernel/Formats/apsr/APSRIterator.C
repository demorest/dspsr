/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/APSRIterator.h"
#include "dsp/Input.h"

void dsp::APSRIterator::init (const Input* input)
{
  unsigned resolution = input->get_resolution();

  set_block_size( input->get_info()->get_nbytes(resolution) );
  set_data_size( get_block_size() / 2 );
  set_increment( input->get_info()->get_nchan() );
}

