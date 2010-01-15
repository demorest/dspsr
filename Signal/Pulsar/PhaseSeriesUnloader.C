/***************************************************************************
 *
 *   Copyright (C) 2003-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PhaseSeriesUnloader.h"
#include "dsp/PhaseSeries.h"
#include "dsp/Operation.h"

#include "Pulsar/Predictor.h"
#include "dirutil.h"
#include "strutil.h"

using namespace std;

//! Constructor
dsp::PhaseSeriesUnloader::PhaseSeriesUnloader ()
{
  extension = ".ar";
  path_add_source = false;
}
    
//! Destructor
dsp::PhaseSeriesUnloader::~PhaseSeriesUnloader ()
{
}

void dsp::PhaseSeriesUnloader::partial (const PhaseSeries* profiles)
{
  if (Operation::verbose)
    cerr << "dsp::PhaseSeriesUnloader::partial unload partial" << endl;
  unload (profiles);
}

void dsp::PhaseSeriesUnloader::finish ()
{
  if (Operation::verbose)
    cerr << "dsp::PhaseSeriesUnloader::finish nothing to do" << endl;
}

//! Set the filename convention
void dsp::PhaseSeriesUnloader::set_convention (FilenameConvention* conv)
{
  convention = conv;
}

dsp::FilenameConvention* dsp::PhaseSeriesUnloader::get_convention ()
{
  return convention;
}

//! Set the prefix to which output data will be written
void dsp::PhaseSeriesUnloader::set_prefix (const std::string& p)
{
  prefix = p;
}

std::string dsp::PhaseSeriesUnloader::get_prefix () const
{
  return prefix;
}

//! Set the path to which output data will be written
void dsp::PhaseSeriesUnloader::set_directory (const std::string& p)
{
  directory = p;
}

std::string dsp::PhaseSeriesUnloader::get_directory () const
{
  return directory;
}
    
//! place output files in a sub-directory named by source
void dsp::PhaseSeriesUnloader::set_path_add_source (bool flag)
{
  path_add_source = flag;
}

bool dsp::PhaseSeriesUnloader::get_path_add_source () const
{
  return path_add_source;
}

//! Set the extension to be added to the end of filenames
void dsp::PhaseSeriesUnloader::set_extension (const std::string& ext)
{
  extension = ext;

  if (ext.empty())
    return;

  // ensure that the first character is a .
  if (extension[0] != '.')
    extension.insert (0, ".");
}


std::string dsp::PhaseSeriesUnloader::get_extension () const
{
  return extension;
}

string dsp::PhaseSeriesUnloader::get_filename (const PhaseSeries* data) const
{
  string the_path;

  if (!directory.empty())
    the_path = directory + "/";

  if (path_add_source)
    the_path += data->get_source() + "/";

  if (!file_is_directory(the_path.c_str()))
    makedir (the_path.c_str());

  return the_path + prefix + convention->get_filename(data) + extension;
}


dsp::FilenameEpoch::FilenameEpoch ()
{
  datestr_pattern = "%Y-%m-%d-%H:%M:%S";
  integer_seconds = 0;
  report_unload = true;
}

void dsp::FilenameEpoch::set_datestr_pattern (const std::string& pattern)
{
  datestr_pattern = pattern;
}

void dsp::FilenameEpoch::set_integer_seconds (unsigned seconds)
{
  integer_seconds = seconds;
}

std::string dsp::FilenameEpoch::get_filename (const PhaseSeries* data) const
{
  MJD epoch = data->get_start_time();

  if (Observation::verbose)
    cerr << "dsp::FilenameEpoch::get_filename epoch=" 
         << epoch.printall() << endl;

  if (integer_seconds)
  {
    // ensure that the epoch is rounded up into the current division
    epoch = data->get_mid_time();

    unsigned seconds = epoch.get_secs();
    unsigned divisions = seconds / integer_seconds;
    epoch = MJD (epoch.intday(), divisions * integer_seconds, 0.0);

    if (Observation::verbose)
      cerr << "dsp::FilenameEpoch::get_filename division start epoch=" 
           << epoch.printall() << endl;
  }

  vector<char> fname (FILENAME_MAX);
  char* filename = &fname[0];

  if (!epoch.datestr( filename, FILENAME_MAX, datestr_pattern.c_str() ))
    throw Error (FailedSys, "dsp::PhaseSeriesUnloader::get_filename",
       "error MJD::datestr(" + datestr_pattern + ")");

  if (report_unload)
    cerr << "unloading " << tostring(data->get_integration_length(),2)
	 << " seconds: " << filename << endl;

  return filename;
}

std::string dsp::FilenamePulse::get_filename (const PhaseSeries* data) const
{
  const Pulsar::Predictor* poly = data->get_folding_predictor();
  if (!poly)
    throw Error (InvalidState, "dsp::FilenamePulse::get_filename",
		 "PhaseSeries does not contain a polyco");

  // add pulse number to the output archive
  Phase phase = poly->phase ( data->get_start_time() );

  if (Observation::verbose)
    cerr << "dsp::FilenamePulse::get_filename phase=" << phase 
	 << " ref=" << data->get_reference_phase() << endl;

  phase = (phase + 0.5 - data->get_reference_phase()).Floor();

  return stringprintf ("pulse_"I64, phase.intturns());
}
