#include "dsp/PMDAQ_Observation.h"

#include "string_utils.h"
#include "genutil.h"
#include "coord.h"

dsp::PMDAQ_Observation::PMDAQ_Observation (const char* header)
{
  if (header == NULL)
    throw_str ("PMDAQ_Observation - no header!");

  // //////////////////////////////////////////////////////////////////////
  //
  // PMDAQ_HEADER_VERSION
  //
  float version;

  int nscanned;

  nscanned = sscanf(&header[10],"%f4.1",&version);
  if (nscanned != 1) {
    throw_str ("PMDAQ_Observation - failed read PMDAQ_HEADER_VERSION");
  }

  //
  // no idea about the size of the data
  //
  ndat = 0;

  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //

  if (strncmp(&header[592],"PARKES",6)==0) {
    set_telescope (7);
  }
      else {
    throw_str ("PMDAQ_Observation - failed to recognise telescope %10.10s\n",
		   &header[592]);
      }

  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  char pmdaq_source_name[13];
  sscanf(&header[528],"%s",pmdaq_source_name);

  set_source (pmdaq_source_name);

  // //////////////////////////////////////////////////////////////////////
  //
  // FREQS, NCHAN ETC.
  //
  double freq_channel_one;
  double chan_incr;

  nscanned = sscanf(&header[208],"%8lf",&chan_incr);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read channel increment\n");
  }

  nscanned = sscanf(&header[224],"%d",&nchan);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read number of channels\n");
  }
  set_nchan ( nchan );
  cerr << "Number of channels is " << nchan << endl;

  nscanned = sscanf(&header[232],"%12lf",&freq_channel_one);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read frequency of channel one\n");
  }

  set_centre_frequency (freq_channel_one - 0.5 * chan_incr + nchan/2 * 
			chan_incr);

  // //////////////////////////////////////////////////////////////////////
  //
  // BW
  //

  set_bandwidth (chan_incr * nchan);   

  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL
  //
  set_npol (1);

  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT
  //
  set_nbit (1);

  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM
  //
  set_state (Signal::Intensity);

  //
  // call this only after setting frequency and telescope
  //
  set_default_basis ();

  // //////////////////////////////////////////////////////////////////////
  //
  // TSAMP
  //
  double sampling_interval;

  nscanned = sscanf(&header[256],"%lf",&sampling_interval);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read sampling interval\n");
  }

  sampling_interval *= 1e-3;

  cerr << " Sampling interval is " << sampling_interval << " secs " << endl;

  set_rate (1.0/sampling_interval);

  // //////////////////////////////////////////////////////////////////////
  //
  // MJD_START
  //

  // MXB MJD is at header[40], UT is at header[48]

  int int_MJD;
  int hh,mm;
  double secs;

  // The approximately one millionth UT to secs parser.

  nscanned = sscanf(&header[40],"%d",&int_MJD);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read integer part of MJD\n");
  }
  nscanned = sscanf(&header[49],"%2d",&hh);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read hours in ut start\n");
  }
  nscanned = sscanf(&header[52],"%2d",&mm);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read minutes in ut start\n%2.2s\n",&header[51]);
  }
  nscanned = sscanf(&header[55],"%9lf",&secs);
  if (nscanned != 1) {
    throw_str("PMDAQ_Observation - failed to read seconds in ut start\n");
  }

  int int_secs = (int) secs;
  double frac_secs = secs - int_secs;

  MJD recording_start_time ((double)int_MJD,
			    (double) (3600.0*hh+60.0*mm+int_secs),
			    frac_secs);

  // //////////////////////////////////////////////////////////////////////
  //
  // OFFSET
  //
  offset_bytes = 0;

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

  // make an identifier name
  set_identifier ("f" + get_default_id());
  set_mode (stringprintf ("%d-bit mode", nbit));
  set_machine ("PMDAQ");

  // //////////////////////////////////////////////////////////////////////
  //
  // RA and DEC
  //
  double ra, dec;
  bool has_position;

  has_position = (str2ra  (&ra,  &header[78]) == 0);
  has_position = (str2dec (&dec, &header[94]) == 0);

  if (!has_position) {
    ra = dec = 0.0;
  }

  coordinates.setRadians(ra, dec);
}




