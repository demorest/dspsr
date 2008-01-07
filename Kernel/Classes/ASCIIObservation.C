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

using namespace std;

dsp::ASCIIObservation::ASCIIObservation (const char* header)
{
  hdr_version = "HDR_VERSION";

  if (header)
    load (header);
}

void dsp::ASCIIObservation::load (const char* header)
{
  if (header == NULL)
    throw Error (InvalidState, "ASCIIObservation::load", "no header!");

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

  //
  // no idea about the size of the data
  //
  set_ndat( 0 );

  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //
  char buffer[64];
  if (ascii_header_get (header, "TELESCOPE", "%s", buffer) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load TELESCOPE");
  
  set_telescope_code( Tempo::code(buffer) );

  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  if (ascii_header_get (header, "SOURCE", "%s", buffer) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load SOURCE");

  set_source (buffer);


  // //////////////////////////////////////////////////////////////////////
  //
  // MODE
  //
  ascii_header_get (header, "MODE", "%s", buffer);
    
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

  if (get_type() == Signal::PolnCal) {

    double calfreq;
    if (ascii_header_get (header, "CALFREQ", "%lf", &calfreq) < 0)
      throw Error (InvalidState, "ASCIIObservation", "failed load FREQ");
    set_calfreq(calfreq);
    cerr << "Set CALFREQ to " << get_calfreq() << endl;

  }

  // //////////////////////////////////////////////////////////////////////
  //
  // FREQ
  //
  double freq;
  if (ascii_header_get (header, "FREQ", "%lf", &freq) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load FREQ");

  set_centre_frequency (freq);

  // //////////////////////////////////////////////////////////////////////
  //
  // BW
  //
  double bw;
  if (ascii_header_get (header, "BW", "%lf", &bw) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load BW");

  set_bandwidth (bw);

  // //////////////////////////////////////////////////////////////////////
  //
  // NCHAN
  //
  int scan_nchan;
  if (ascii_header_get (header, "NCHAN", "%d", &scan_nchan) < 0)
    set_nchan (1);
  else
    set_nchan (scan_nchan);

  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL
  //
  int scan_npol;
  if (ascii_header_get (header, "NPOL", "%d", &scan_npol) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load NPOL");

  set_npol (scan_npol);

  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT
  //
  int scan_nbit;
  if (ascii_header_get (header, "NBIT", "%d", &scan_nbit) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load NBIT");

  set_nbit (scan_nbit);

  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM
  //
  int scan_ndim;
  if (ascii_header_get (header, "NDIM", "%d", &scan_ndim) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load NDIM");

  set_ndim(scan_ndim);

  switch (scan_ndim) {
  case 1:
    set_state (Signal::Nyquist); break;
  case 2:
    set_state (Signal::Analytic); break;
  default:
    throw Error (InvalidState, "ASCIIObservation",
		 "invalid NDIM=%d\n", scan_ndim);
  }
  
  //
  // call this only after setting frequency and telescope
  //
  set_default_basis ();


  // //////////////////////////////////////////////////////////////////////
  //
  // TSAMP
  //
  double sampling_interval;
  if (ascii_header_get (header, "TSAMP", "%lf", &sampling_interval)<0)
    throw Error (InvalidState, "ASCIIObservation", "failed load TSAMP");

  /* IMPORTANT: TSAMP is the sampling period in microseconds */
  sampling_interval *= 1e-6;

  set_rate (1.0/sampling_interval);

  // //////////////////////////////////////////////////////////////////////
  //
  // UTC_START
  //
  if (ascii_header_get (header, "UTC_START", "%s", buffer) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load UTC_START");

  struct tm utc;
  if (strptime (buffer, "%Y-%m-%d-%H:%M:%S", &utc) == NULL)
    throw Error (InvalidState, "ASCIIObservation",
		 "failed strptime (%s)", buffer);

  MJD recording_start_time( timegm (&utc) );

#ifdef _DEBUG
  if (ascii_header_get (header, "MJD_START", "%s", buffer) >= 0)
    cerr << "MJD_START=" << buffer
	 << " MJD(UTC)=" << recording_start_time.printdays(13) << endl;
#endif

  // //////////////////////////////////////////////////////////////////////
  //
  // OBS_OFFSET
  //
  offset_bytes = 0;
  if (ascii_header_get (header, "OBS_OFFSET", UI64, &offset_bytes) < 0 &&
      ascii_header_get (header, "OFFSET", UI64, &offset_bytes) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed load OBS_OFFSET");

  // //////////////////////////////////////////////////////////////////////
  //
  // CALCULATE the various offsets and sizes
  //
  
  double offset_seconds = get_nsamples(offset_bytes) * sampling_interval;
  set_start_time (recording_start_time + offset_seconds);

  //
  // until otherwise, the band is centred on the centre frequency
  //
  dc_centred = true;

  // //////////////////////////////////////////////////////////////////////
  //
  // INSTRUMENT
  //
  if (ascii_header_get (header, "INSTRUMENT", "%s", buffer) == 1)
    set_machine (buffer);

  // make an identifier name
  set_identifier (get_default_id());

  // //////////////////////////////////////////////////////////////////////
  //
  // RA and DEC
  //
  bool has_position = true;
  double ra, dec;

  if (has_position)
    has_position = (ascii_header_get (header, "RA", "%s", buffer) == 1);

  if (has_position)
    has_position = (str2ra (&ra, buffer) == 0);

  if (has_position)
    has_position = (ascii_header_get (header, "DEC", "%s", buffer) == 1);

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
  float version;
  if (ascii_header_set (header, hdr_version.c_str(), "%f", version) < 0)
    cerr << "ASCIIObservation: failed unload " << hdr_version << endl;


  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //
  if (ascii_header_set (header, "TELESCOPE", "%c", get_telescope_code() ) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload TELESCOPE");


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
  uint64 offset_bytes = get_nbytes( (uint64)offset_samples );

  if (ascii_header_set (header, "OBS_OFFSET", UI64, offset_bytes) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload OBS_OFFSET");


  // //////////////////////////////////////////////////////////////////////
  //
  // INSTRUMENT
  //
  if (ascii_header_set (header, "INSTRUMENT", "%s", get_machine().c_str()) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed unload INSTRUMENT");

}

