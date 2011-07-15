/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FilterbankConfig.h"

using dsp::Filterbank;

Filterbank::Config::Config ()
{
  nchan = 1;
  when = After;  // not good, but the original default
}

std::ostream& operator << (std::ostream& os, const Filterbank::Config& config)
{
  os << config.get_nchan();
  if (config.get_dedisperse_when() == Filterbank::Config::Before)
    os << ":B";
  else if (config.get_dedisperse_when() == Filterbank::Config::During)
    os << ":D";
  return os;
}

//! Extraction operator
std::istream& operator >> (std::istream& is, Filterbank::Config& config)
{
  unsigned value;
  is >> value;

  config.set_nchan (value);
  config.set_dedisperse_when (Filterbank::Config::After);

  if (!is)
    return is;

  std::streampos pos = is.tellg();

  std::string when;
  is >> when;

  if (when == ":D" || when == ":d")
    config.set_dedisperse_when (Filterbank::Config::During);
  else if (when == ":B" || when == ":b")
    config.set_dedisperse_when (Filterbank::Config::Before);
  else if (when.length())
    is.seekg (pos);

  return is;
}

