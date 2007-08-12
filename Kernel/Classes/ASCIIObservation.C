/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ASCIIObservation.h"
#include "Telescope.h"
#include "strutil.h"
#include "coord.h"

#include "ascii_header.h"

using namespace std;

dsp::ASCIIObservation::ASCIIObservation (const char* header)
{
  hdr_version = "HDR_VERSION";
  obs_offset = "OBS_OFFSET";

  parse (header);
}

void dsp::ASCIIObservation::parse (const char* header)
{
  if (header == NULL)
    throw Error (InvalidState, "ASCIIObservation", "no header!");

  // //////////////////////////////////////////////////////////////////////
  //
  // HDR_VERSION
  //
  float version;
  if (ascii_header_get (header, hdr_version.c_str(), "%f", &version) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read "+hdr_version);

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
    throw Error (InvalidState, "ASCIIObservation", "failed read TELESCOPE");
  
  set_telescope_code( Telescope::code(buffer) );

  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  if (ascii_header_get (header, "SOURCE", "%s", buffer) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read SOURCE");

  set_source (buffer);


  // //////////////////////////////////////////////////////////////////////
  //
  // MODE
  //
  ascii_header_get(header, "MODE", "%s", buffer);
    
  if (strncmp(buffer,"PSR",3) == 0) {
    cerr << "Source is Pulsar" << endl;
    set_type(Signal::Pulsar);
  }
  else if (strncmp(buffer,"CAL",3) == 0) {
    cerr << "Source is CAL" << endl;
    set_type(Signal::PolnCal);
  }
  else if (strncmp(buffer,"LEV",3) == 0) {
    cerr << "Source is LEVCAL" << endl;
    set_type(Signal::PolnCal);
  }
  else {
    cerr << "ASCIIObservation: unknown MODE, assuming Pulsar" << endl;
    set_type(Signal::Pulsar);
  }

  if (get_type() == Signal::PolnCal) {

    double calfreq;
    if (ascii_header_get (header, "CALFREQ", "%lf", &calfreq) < 0)
      throw Error (InvalidState, "ASCIIObservation", "failed read FREQ");
    set_calfreq(calfreq);
    cerr << "Set CALFREQ to " << get_calfreq() << endl;

  }

  // //////////////////////////////////////////////////////////////////////
  //
  // FREQ
  //
  double freq;
  if (ascii_header_get (header, "FREQ", "%lf", &freq) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read FREQ");

  set_centre_frequency (freq);

  // //////////////////////////////////////////////////////////////////////
  //
  // BW
  //
  double bw;
  if (ascii_header_get (header, "BW", "%lf", &bw) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read BW");

  set_bandwidth (bw);

  //
  // CPSR2 data is single-channel unless this is a detected CPSR2 FB see later
  //
  set_nchan(1);

  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL
  //
  int scan_npol;
  if (ascii_header_get (header, "NPOL", "%d", &scan_npol) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read NPOL");

  set_npol (scan_npol);

  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT
  //
  int scan_nbit;
  if (ascii_header_get (header, "NBIT", "%d", &scan_nbit) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read NBIT");

  set_nbit (scan_nbit);

  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM
  //
  int scan_ndim;
  if (ascii_header_get (header, "NDIM", "%d", &scan_ndim) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read NDIM");
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
    throw Error (InvalidState, "ASCIIObservation", "failed read TSAMP");

  /* IMPORTANT: TSAMP is the sampling period in microseconds */
  sampling_interval *= 1e-6;

  set_rate (1.0/sampling_interval);

  // //////////////////////////////////////////////////////////////////////
  //
  // MJD_START
  //
  if (ascii_header_get (header, "MJD_START", "%s", buffer) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read MJD_START");

  MJD recording_start_time (buffer);

  // //////////////////////////////////////////////////////////////////////
  //
  // OBS_OFFSET
  //
  offset_bytes = 0;
  if (ascii_header_get (header, obs_offset.c_str(), UI64, &offset_bytes) < 0)
    throw Error (InvalidState, "ASCIIObservation", "failed read "+obs_offset);



  // //////////////////////////////////////////////////////////////////////
  //
  // CALCULATE the various offsets and sizes
  //
  uint64 bitsperbyte = 8;
  uint64 bitspersample = get_nbit()*get_npol();

  uint64 offset_samples = offset_bytes * bitsperbyte / bitspersample;
  
  double offset_seconds = double(offset_samples) * sampling_interval;

  set_start_time (recording_start_time + offset_seconds);

  //
  // until otherwise, the band is centred on the centre frequency
  //
  dc_centred = true;

  // //////////////////////////////////////////////////////////////////////
  //
  // INSTRUMENT
  //
  if (ascii_header_get (header, "INSTRUMENT", "%s", buffer) == 0)
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

