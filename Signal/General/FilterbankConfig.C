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
  freq_res = 0;  // unspecified
  when = After;  // not good, but the original default
}

std::ostream& dsp::operator << (std::ostream& os,
				const Filterbank::Config& config)
{
  os << config.get_nchan();
  if (config.get_convolve_when() == Filterbank::Config::Before)
    os << ":B";
  else if (config.get_convolve_when() == Filterbank::Config::During)
    os << ":D";
  else if (config.get_freq_res() != 1)
    os << ":" << config.get_freq_res();

  return os;
}

//! Extraction operator
std::istream& dsp::operator >> (std::istream& is, Filterbank::Config& config)
{
  unsigned value;
  is >> value;

  config.set_nchan (value);
  config.set_convolve_when (Filterbank::Config::After);

  if (!is || is.peek() != ':')
    return is;

  // throw away the colon
  is.get();

  if (is.peek() == 'D' || is.peek() == 'd')
  {
    is.get();  // throw away the D
    config.set_convolve_when (Filterbank::Config::During);
  }
  else if (is.peek() == 'B' || is.peek() == 'b')
  {
    is.get();  // throw away the B
    config.set_convolve_when (Filterbank::Config::Before);
  }
  else
  {
    unsigned nfft;
    is >> nfft;
    config.set_freq_res (nfft);
  }

  return is;
}

