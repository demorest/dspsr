/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CloneArchive.h"
#include "Pulsar/Archive.h"

using namespace std;

dsp::CloneArchive::CloneArchive (const Pulsar::Archive* archive)
  : OutputArchive ("CloneArchive")
{
  instance = archive;
}

dsp::CloneArchive::CloneArchive (const CloneArchive& copy)
  : OutputArchive ("CloneArchive")
{
  instance = copy.instance;
}

dsp::CloneArchive::~CloneArchive()
{
}

dsp::dspExtension* dsp::CloneArchive::clone() const
{
  return new CloneArchive (*this);
}

//! Return a clone of the Pulsar::Archive instance
Pulsar::Archive* dsp::CloneArchive::new_Archive () const
{
  return instance->clone ();
}

