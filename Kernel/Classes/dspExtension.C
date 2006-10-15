/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/dspExtension.h"

using namespace std;

//! Constructor
dsp::dspExtension::dspExtension(string _name, bool _can_only_have_one)
{
  name = _name;
  can_only_have_one = _can_only_have_one;
}

//! Virtual destructor
dsp::dspExtension::~dspExtension()
{
}

//! If true, then you can only have one of this type of dspExtension per Observation instantiation
bool dsp::dspExtension::must_only_have_one() const 
{
  return can_only_have_one; 
}
