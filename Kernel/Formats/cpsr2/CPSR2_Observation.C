#include "dsp/CPSR2_Observation.h"
#include "cpsr2_header.h"

#include "string_utils.h"
#include "genutil.h"
#include "coord.h"

dsp::CPSR2_Observation::CPSR2_Observation (const char* header)
{
  if (header == NULL)
    throw_str ("CPSR2_Observation - no header!");

  // //////////////////////////////////////////////////////////////////////
  //
  // CPSR2_HEADER_VERSION
  //
  float version;
  if (ascii_header_get (header, 
			"CPSR2_HEADER_VERSION", "%f", &version) < 0)
    throw_str ("CPSR2_Observation - failed read CPSR2_HEADER_VERSION");

  //
  // no idea about the size of the data
  //
  ndat = 0;

  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //
  char hdrstr[64];
  if (ascii_header_get (header, "TELESCOPE", "%s", hdrstr) < 0)
    throw_str ("CPSR2_Observation - failed read TELESCOPE");

  string tel = hdrstr;
  if ( !strcasecmp (hdrstr, "parkes") || tel == "PKS") 
    set_telescope_code (7);
  else if ( !strcasecmp (hdrstr, "GBT") || tel == "GBT")
    set_telescope_code (1);
  else {
    cerr << "CPSR2_Observation:: Warning using code" << hdrstr[0] << endl;
    set_telescope_code (hdrstr[0]);
  }
  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  if (ascii_header_get (header, "SOURCE", "%s", hdrstr) < 0)
    throw_str ("CPSR2_Observation - failed read SOURCE");

  set_source (hdrstr);

  // //////////////////////////////////////////////////////////////////////
  //
  // FREQ
  //
  double freq;
  if (ascii_header_get (header, "FREQ", "%lf", &freq) < 0)
    throw_str ("CPSR2_Observation - failed read FREQ");

  set_centre_frequency (freq);

  // //////////////////////////////////////////////////////////////////////
  //
  // BW
  //
  double bw;
  if (ascii_header_get (header, "BW", "%lf", &bw) < 0)
    throw_str ("CPSR2_Observation - failed read BW");

  set_bandwidth (bw);

  //
  // CPSR2 data is single-channel
  //
  set_nchan(1);

  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL
  //
  int npol;
  if (ascii_header_get (header, "NPOL", "%d", &npol) < 0)
    throw_str ("CPSR2_Observation - failed read NPOL");

  set_npol (npol);

  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT
  //
  int nbit;
  if (ascii_header_get (header, "NBIT", "%d", &nbit) < 0)
    throw_str ("CPSR2_Observation - failed read NBIT");

  set_nbit (nbit);

  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM
  //
  int ndim;
  if (ascii_header_get (header, "NDIM", "%d", &ndim) < 0)
    throw_str ("CPSR2_Observation - failed read NDIM");

  switch (ndim) {
  case 1:
    set_state (Signal::Nyquist); break;
  case 2:
    set_state (Signal::Analytic); break;
  default:
    throw_str ("CPSR2_Observation - invalid NDIM=%d\n", ndim);
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
    throw_str ("CPSR2_Observation - failed read TSAMP");

  /* IMPORTANT: TSAMP is the sampling period in microseconds */
  sampling_interval *= 1e-6;

  set_rate (1.0/sampling_interval);

  // //////////////////////////////////////////////////////////////////////
  //
  // MJD_START
  //
  if (ascii_header_get (header, "MJD_START", "%s", hdrstr) < 0)
    throw_str ("CPSR2_Observation - failed read MJD_START");

  MJD recording_start_time (hdrstr);

  // //////////////////////////////////////////////////////////////////////
  //
  // OFFSET
  //
  offset_bytes = 0;
  if (ascii_header_get (header, "OFFSET", UI64, &offset_bytes) < 0)
    throw_str ("CPSR2_Observation - failed read OFFSET");

  if (version < 0.2) {
    // //////////////////////////////////////////////////////////////////////
    //
    // NMBYTES
    //
    uint64 offset_Mbytes = 0;
    if (ascii_header_get (header, "NMBYTES", UI64, &offset_Mbytes) < 0)
      cerr << "CPSR2_Observation - no NMBYTES...  assuming local" << endl;

    cerr << "CPSR2_HEADER_VERSION 0.1 offset MBytes " << offset_Mbytes << endl;

    uint64 MByte = 1024 * 1024;
    offset_bytes += offset_Mbytes * MByte;
  }


  // //////////////////////////////////////////////////////////////////////
  //
  // CALCULATE the various offsets and sizes
  //
  uint64 bitsperbyte = 8;
  uint64 bitspersample = nbit*npol;

  uint64 offset_samples = offset_bytes * bitsperbyte / bitspersample;
  
  double offset_seconds = double(offset_samples) * sampling_interval;

  set_start_time (recording_start_time + offset_seconds);

  //
  // until otherwise, the band is centred on the centre frequency
  //
  dc_centred = true;

  // //////////////////////////////////////////////////////////////////////
  //
  // PRIMARY
  //
  if (ascii_header_get (header, "PRIMARY", "%s", hdrstr) < 0)
    throw_str ("CPSR2_Observation - failed read PRIMARY");

  string primary = hdrstr;
  string prefix = "u";

  if (primary == "cpsr1")
    prefix = "m";
  if (primary == "cpsr2")
    prefix = "n";

  if (primary == "cgsr1")
    prefix = "p";
  if (primary == "cgsr2")
    prefix = "q";

  // make an identifier name
  set_identifier (prefix + get_default_id());
  set_mode (stringprintf ("%d-bit mode", nbit));
  set_machine ("CPSR2");

  // //////////////////////////////////////////////////////////////////////
  //
  // RA and DEC
  //
  bool has_position = true;
  double ra, dec;

  if (has_position)
    has_position = (ascii_header_get (header, "RA", "%s", hdrstr) == 0);
  if (has_position)
    has_position = (str2ra (&ra, hdrstr) == 0);
  if (has_position)
    has_position = (ascii_header_get (header, "DEC", "%s", hdrstr) == 0);
  if (has_position)
    has_position = (str2dec (&dec, hdrstr) == 0);

  if (!has_position) {
    ra = dec = 0.0;
  }

  coordinates.setRadians(ra, dec);
}

