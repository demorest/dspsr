/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ASCIIObservation.h"
#include "strutil.h"
#include "coord.h"
#include "tempo++.h"

#include "ascii_header.h"

#include <string.h>

#include <algorithm>

using namespace std;

dsp::ASCIIObservation::ASCIIObservation (const char* header)
{
  hdr_version = "HDR_VERSION";

  // The default-required keywords
  required_keys.clear();
  required_keys.push_back("TELESCOPE");
  required_keys.push_back("SOURCE");
  required_keys.push_back("CALFREQ");
  required_keys.push_back("FREQ");
  required_keys.push_back("BW");
  required_keys.push_back("NPOL");
  required_keys.push_back("NBIT");
  // NDIM and NCHAN have a default value of 1
  required_keys.push_back("TSAMP");
  required_keys.push_back("UTC_START");
  required_keys.push_back("OBS_OFFSET");

  if (header)
    load (header);
}

bool dsp::ASCIIObservation::is_required (std::string key)
{
  if ( count(required_keys.begin(), required_keys.end(), key) > 0 )
    return true;
  else
    return false;
}

void dsp::ASCIIObservation::set_required (std::string key,
    bool required)
{

  if ( required && is_required(key))
    return;

  if ( !required && !is_required(key) )
    return;

  if (required)
  {
    required_keys.push_back(key);
  }

  else
  {
    std::vector< std::string >::iterator it;
    it = remove(required_keys.begin(), required_keys.end(), key);
    required_keys.erase(it, required_keys.end());
  }

}

void dsp::ASCIIObservation::load (const char* header)
{
  if (header == NULL)
    throw Error (InvalidState, "ASCIIObservation::load", "no header!");

  if (verbose)
  {
    cerr << "ASCIIObservation::load required keywords:" << endl;
    for (unsigned i=0; i<required_keys.size(); i++)
      cerr << "  " << required_keys[i] << endl;
  }


  // //////////////////////////////////////////////////////////////////////
  //
  // HDR_VERSION
  //
  float version;
  if (ascii_header_get (header, hdr_version.c_str(), "%f", &version) < 0)
  {
    /* Provide backward-compatibility with CPSR2 header */
    if (ascii_header_get (header, "CPSR2_HEADER_VERSION", "%f", &version) < 0)
      cerr << "ASCIIObservation: failed read " << hdr_version << endl;
    else
      set_machine ("CPSR2");
  }

  char buffer[64];

  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //
  // Note: The function ascii_header_check will check the list of 
  // required keywords and throw an error only if the requested
  // keyword is required and not present.
  // If a value < 0 is returned here, then the keyword
  // has been marked as not required so rather than throwing an error, 
  // some default/unknown value should be used.
  //
  if (ascii_header_check (header, "TELESCOPE", "%s", buffer) < 0)
    set_telescope ("unknown");
  else
    set_telescope (buffer);

  // //////////////////////////////////////////////////////////////////////
  //
  // RECEIVER
  //
  if (ascii_header_check (header, "RECEIVER", "%s", buffer) < 0)
    set_receiver ("unknown");
  else
    set_receiver (buffer);

  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  if (ascii_header_check (header, "SOURCE", "%s", buffer) < 0)
    set_source ("unknown");
  else
    set_source (buffer);

  // //////////////////////////////////////////////////////////////////////
  //
  // MODE
  //
  ascii_header_check (header, "MODE", "%s", buffer);
    
  if (strncmp(buffer,"PSR",3) == 0)
    set_type(Signal::Pulsar);
  else if (strncmp(buffer,"CAL",3) == 0)
    set_type(Signal::PolnCal);
  else if (strncmp(buffer,"LEV",3) == 0)
    set_type(Signal::PolnCal);
  else {
    if (verbose)
      cerr << "ASCIIObservation: unknown MODE, assuming Pulsar" << endl;
    set_type(Signal::Pulsar);
  }

  if (get_type() == Signal::PolnCal)
  {
    double calfreq;
    if (ascii_header_check (header, "CALFREQ", "%lf", &calfreq) < 0)
      set_calfreq(0.0);
    else
      set_calfreq(calfreq);
  }

  // //////////////////////////////////////////////////////////////////////
  //
  // FREQ
  //
  double freq;
  if (ascii_header_check (header, "FREQ", "%lf", &freq) < 0)
    set_centre_frequency (0.0);
  else
    set_centre_frequency (freq);

  // //////////////////////////////////////////////////////////////////////
  //
  // BW
  //
  double bw;
  if (ascii_header_check (header, "BW", "%lf", &bw) < 0)
    set_bandwidth (0.0);
  else
    set_bandwidth (bw);

  // //////////////////////////////////////////////////////////////////////
  //
  // NCHAN
  //
  int scan_nchan;
  if (ascii_header_check (header, "NCHAN", "%d", &scan_nchan) < 0)
    set_nchan (1);
  else
    set_nchan (scan_nchan);

  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL
  //
  int scan_npol;
  if (ascii_header_check (header, "NPOL", "%d", &scan_npol) < 0)
    set_npol (1);
  else
    set_npol (scan_npol);

  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT
  //
  int scan_nbit;
  if (ascii_header_check (header, "NBIT", "%d", &scan_nbit) < 0)
    set_nbit (2);
  else
    set_nbit (scan_nbit);


  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM
  //
  int scan_ndim;
  if (ascii_header_check (header, "NDIM", "%d", &scan_ndim) < 0)
    set_ndim (1);
  else
    set_ndim (scan_ndim);

  switch (scan_ndim)
  {
  case 1:
    set_state (Signal::Nyquist); break;
  case 2:
    set_state (Signal::Analytic); break;
  default:
    throw Error (InvalidState, "ASCIIObservation",
		 "invalid NDIM=%d\n", scan_ndim);
  }

  // //////////////////////////////////////////////////////////////////////
  //
  // NDAT
  //

  /*
    This parameter is optional because it can be unknown, as in the case
    of reading data from a ring buffer, or determined from the size of
    the file, in which case it is a function of NCHAN, NPOL, NDIM, NBIT.
  */

  uint64_t scan_ndat = 0;
  if (ascii_header_check (header, "NDAT", "%"PRIu64, &scan_ndat) >= 0)
    set_ndat( scan_ndat );
  else
    set_ndat( 0 );

  if (verbose)
    cerr << "dsp::ASCIIObservation::load ndat=" << scan_ndat << endl;

  // //////////////////////////////////////////////////////////////////////
  //
  // STATE
  //
  if (ascii_header_check (header, "STATE", "%s", buffer) >= 0)
  {
    if (verbose)
      cerr << "dsp::ASCIIObservation::load STATE=" << buffer << endl;
    set_state( Signal::string2State(buffer) );
  }

  int scan_dsb;
  if (ascii_header_check (header, "DSB", "%d", &scan_dsb) >= 0)
    set_dual_sideband (scan_dsb == 1);

  // //////////////////////////////////////////////////////////////////////
  //
  // TSAMP
  //
  double sampling_interval=0.0;
  ascii_header_check (header, "TSAMP", "%lf", &sampling_interval);

  /* IMPORTANT: TSAMP is the sampling period in microseconds */
  sampling_interval *= 1e-6;

  set_rate (1.0/sampling_interval);

  // //////////////////////////////////////////////////////////////////////
  //
  // UTC_START
  //
  MJD recording_start_time( MJD::zero );
  if (ascii_header_check (header, "UTC_START", "%s", buffer) > 0)
  {

    if (verbose)
      cerr << "dsp::ASCIIObservation::load UTC_START='" 
        << buffer << "'" << endl;

    struct tm utc;
    if (strptime (buffer, "%Y-%m-%d-%H:%M:%S", &utc) == NULL)
      throw Error (InvalidState, "ASCIIObservation",
                   "failed strptime (%s)", buffer);

    if (verbose)
      cerr << "dsp::ASCIIObservation::load asctime=" << asctime (&utc) << endl;

    recording_start_time = MJD( timegm (&utc) );

#ifdef _DEBUG
    if (ascii_header_check (header, "MJD_START", "%s", buffer) >= 0)
      cerr << "MJD_START=" << buffer
           << " MJD(UTC)=" << recording_start_time.printdays(13) << endl;
#endif

    if (verbose)
      cerr << "dsp::ASCIIObservation::load start_mjd=" 
           << recording_start_time << endl;
  }

  // //////////////////////////////////////////////////////////////////////
  //
  // OBS_OFFSET
  //
  offset_bytes = 0;
  try
  {
    ascii_header_check (header, "OBS_OFFSET", UI64, &offset_bytes);
  }
  catch (Error &error)
  {
    if (ascii_header_check (header, "OFFSET", UI64, &offset_bytes) < 0)
      throw error;
  }

  // //////////////////////////////////////////////////////////////////////
  //
  // CALCULATE the various offsets and sizes
  //
  
  if ( recording_start_time != MJD::zero )
  {
    double offset_seconds = get_nsamples(offset_bytes) * sampling_interval;
    set_start_time (recording_start_time + offset_seconds);
  }

  // //////////////////////////////////////////////////////////////////////
  //
  // INSTRUMENT
  //
  if (ascii_header_check (header, "INSTRUMENT", "%s", buffer) == 1)
    set_machine (buffer);

  // //////////////////////////////////////////////////////////////////////
  //
  // RA and DEC
  //
  bool has_position = true;
  double ra, dec;

  if (has_position)
    has_position = (ascii_header_check (header, "RA", "%s", buffer) == 1);

  if (has_position)
    has_position = (str2ra (&ra, buffer) == 0);

  if (has_position)
    has_position = (ascii_header_check (header, "DEC", "%s", buffer) == 1);

  if (has_position)
    has_position = (str2dec2 (&dec, buffer) == 0);

  if (!has_position)
    ra = dec = 0.0;

  coordinates.setRadians (ra, dec);
}

/* ***********************************************************************
 *
 *
 *
 * *********************************************************************** */

void dsp::ASCIIObservation::unload (char* header)
{
  if (header == NULL)
    throw Error (InvalidState, "ASCIIObservation::unload", "no header!");


  // //////////////////////////////////////////////////////////////////////
  //
  // HDR_VERSION
  //
  float version = 0.0;
  if (ascii_header_set (header, hdr_version.c_str(), "%f", version) < 0)
    cerr << "ASCIIObservation: failed unload " << hdr_version << endl;


  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //
  if (ascii_header_set (header, "TELESCOPE", "%s", 
			get_telescope().c_str() ) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload TELESCOPE");


  // //////////////////////////////////////////////////////////////////////
  //
  // RECEIVER
  //
  if (ascii_header_set (header, "RECEIVER", "%s", get_receiver().c_str()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload RECEIVER");


  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  if (ascii_header_set (header, "SOURCE", "%s", get_source().c_str()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload SOURCE");


  // //////////////////////////////////////////////////////////////////////
  //
  // MODE
  //
  string mode;
  switch (get_type())
  {
  case Signal::Pulsar: mode = "PSR"; break;
  case Signal::PolnCal: mode = "CAL"; break;
  default: mode = "UNKNOWN"; break;
  }
  ascii_header_set (header, "MODE", "%s", mode.c_str());
    

  if (get_type() == Signal::PolnCal
      && ascii_header_set (header, "CALFREQ", "%lf", get_calfreq()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload FREQ");


  // //////////////////////////////////////////////////////////////////////
  //
  // FREQ
  //
  if (ascii_header_set (header, "FREQ", "%lf", get_centre_frequency()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload FREQ");


  // //////////////////////////////////////////////////////////////////////
  //
  // BW
  //
  if (ascii_header_set (header, "BW", "%lf", get_bandwidth()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload BW");


  // //////////////////////////////////////////////////////////////////////
  //
  // NCHAN
  //
  if (ascii_header_set (header, "NCHAN", "%d", get_nchan()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload NCHAN");


  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL
  //
  if (ascii_header_set (header, "NPOL", "%d", get_npol()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload NPOL");


  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT
  //
  if (ascii_header_set (header, "NBIT", "%d", get_nbit()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload NBIT");


  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM
  //
  if (ascii_header_set (header, "NDIM", "%d", get_ndim()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload NDIM");


  // //////////////////////////////////////////////////////////////////////
  //
  // TSAMP
  //
  /* IMPORTANT: TSAMP is the sampling period in microseconds */

  double sampling_interval = 1e6 / get_rate();
  if (ascii_header_set (header, "TSAMP", "%lf", sampling_interval)<0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload TSAMP");


  // //////////////////////////////////////////////////////////////////////
  //
  // UTC_START
  //
  MJD epoch = get_start_time();
  MJD integer_second ( epoch.intday(), epoch.get_secs(), 0.0 );

  char datestr [64];
  integer_second.datestr( datestr, 64, "%Y-%m-%d-%H:%M:%S" );

  if (ascii_header_set (header, "UTC_START", "%s", datestr) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload UTC_START");

  // //////////////////////////////////////////////////////////////////////
  //
  // OBS_OFFSET
  //
  double offset_samples = epoch.get_fracsec() * get_rate();
  uint64_t offset_bytes = get_nbytes( (uint64_t)offset_samples );

  if (ascii_header_set (header, "OBS_OFFSET", UI64, offset_bytes) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload OBS_OFFSET");


  // //////////////////////////////////////////////////////////////////////
  //
  // INSTRUMENT
  //
  if (ascii_header_set (header, "INSTRUMENT", "%s", get_machine().c_str()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload INSTRUMENT");

}

