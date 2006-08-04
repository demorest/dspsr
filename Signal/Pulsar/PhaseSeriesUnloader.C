/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/PhaseSeriesUnloader.h"
#include "dsp/PhaseSeries.h"
#include "polyco.h"
#include "string_utils.h"

//! Constructor
dsp::PhaseSeriesUnloader::PhaseSeriesUnloader ()
{
  filename_extension = ".ar";
  force_filename = false;
  source_filename = false;
}
    
//! Destructor
dsp::PhaseSeriesUnloader::~PhaseSeriesUnloader ()
{
}

string dsp::PhaseSeriesUnloader::get_filename (const PhaseSeries* data) const
{
  if ( data->get_archive_filename() != string() )
    return make_unique(data->get_archive_filename(),"",data);

  string filename;
  string fname_extension = filename_extension;

  if ( data->get_archive_filename_extension() != string() )
    fname_extension = data->get_archive_filename_extension();

  if ( filename_pattern.empty() ){
    filename = data->get_default_id() + fname_extension;
  }
  else {
    char* fname = new char[FILENAME_MAX];
    char* retval = data->get_start_time().datestr ( fname, FILENAME_MAX,
						    filename_pattern.c_str() );

    if (retval)
      filename = retval;

    delete[] fname;

    cerr << "filename = " << filename << endl;

    if (!retval)
      throw Error (FailedSys, "dsp::PhaseSeriesUnloader::get_filename",
		   "error MJD::datestr(" + filename_pattern + ")");
  }

  return make_unique(filename,fname_extension,data);
}

string dsp::PhaseSeriesUnloader::make_unique (const string& filename,
					      const string& fname_extension,
					      const PhaseSeries* data) const
{
  string path;

  if (!filename_path.empty())
    path = filename_path + "/";

  if (source_filename)
    path += data->get_source() + "/";

  if (!this_file_exists(path.c_str()))
    makedir (path.c_str());

  string unique_filename = filename;

  if (Observation::verbose)
    cerr << "dsp::PhaseSeriesUnloader::make_unique filename=" << filename
	 << " ext=" << fname_extension << " force=" << force_filename 
	 << " length=" << data->get_integration_length() << endl;

  if (force_filename || data->get_integration_length() > 1.0)
    return path + unique_filename;

  // small files need a more unique filename

  const polyco* poly = data->get_folding_polyco();
  if (poly) {

    // add pulse number to the output archive
    Phase phase = poly->phase ( data->get_start_time() );

    if (Observation::verbose)
      cerr << "dsp::PhaseSeriesUnloader::make_unique phase=" << phase 
	   << " ref=" << data->get_reference_phase() << endl;

    phase = (phase + 0.5-data->get_reference_phase()).Floor();
    
    unique_filename = stringprintf ("pulse_"I64, phase.intturns());
    unique_filename += fname_extension;

  }
  else
    cerr << "WARNING: integration length < 1 sec.\n"
      "'" << unique_filename << "' may not be unique." << endl;
  
  return path + unique_filename;
}

//! Set the PhaseSeries from which Profile data will be constructed
void dsp::PhaseSeriesUnloader::set_profiles (const PhaseSeries* _profiles)
{
  profiles = _profiles;
}

/*! If this method is called, then set_extension is ignored.  The filename
  may contain date and time format conversion specifiers as described by
  the strftime man page */
void dsp::PhaseSeriesUnloader::set_filename (const char* filename)
{
  filename_pattern = filename;
}

//! Set the extension to be used by get_filename
void dsp::PhaseSeriesUnloader::set_extension (const char* extension)
{
  filename_extension = extension;
}

