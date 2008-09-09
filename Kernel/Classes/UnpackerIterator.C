/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/UnpackerIterator.h"

dsp::Unpacker::Iterator::Iterator (Implementation* _impl)
{
  impl = _impl;
}

dsp::Unpacker::Iterator::~Iterator ()
{
  if (impl) delete impl; impl = 0;
}

//! Dereferencing operator
unsigned char dsp::Unpacker::Iterator::operator * () const
{
  return impl->get_value();
}

//! Increment operator
void dsp::Unpacker::Iterator::operator ++ ()
{
  impl->increment();
}

//! Comparison operator
bool dsp::Unpacker::Iterator::operator < (const unsigned char* ptr)
{
  return impl->less_than (ptr);
}
