#include <string>

#include "dsp/PMDAQ_Observation.h"

#include "string_utils.h"
#include "genutil.h"
#include "coord.h"

// See /psr/cvshome/hknight/soft_swin/search/sc_td/sc_pmhead.inc for more details on header format

dsp::PMDAQ_Observation::PMDAQ_Observation(const char* header) : Observation()
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
  set_ndat( 0 );

  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //

  if (strncmp(&header[592],"PARKES",6)==0)
    set_telescope_code (7);
  else
    throw_str ("PMDAQ_Observation - failed to recognise telescope %10.10s\n",
	       &header[592]);

  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  char pmdaq_source_name[13];
  sscanf(&header[528],"%s",pmdaq_source_name);

  set_source (pmdaq_source_name);

  // //////////////////////////////////////////////////////////////////////
  //
  // nchan(s), centre_frequency(s), bandwidth(s) 
  //
  unsigned nfilters = read_header<unsigned>(header,206,2);

  double chanbw1 = read_header<double>(header,208,8);
  double chanbw2 = read_header<double>(header,216,8);

  unsigned nchan1 = read_header<unsigned>(header,224,4);
  unsigned nchan2 = read_header<unsigned>(header,228,4);

  double freq_chan1_1 = read_header<double>(header,232,12);
  double freq_chan1_2 = read_header<double>(header,244,12);

  if( nfilters==1 ){
    set_nchan( nchan1 );

    if( nchan==512 && fabs(chanbw1) > 2.9 ){
      fprintf(stderr,"WARNING: PMDAQ_Observation has read in that nchan=512 and chanbw1=%f which seems strange.  Taking the guess that chanbw1 should be -0.5\n",
	      chanbw1);
      chanbw1 = -0.5;
    }

    set_centre_frequency (freq_chan1_1 - 0.5 * chanbw1 + get_nchan()/2 * chanbw1);

    set_bandwidth (chanbw1 * get_nchan());

    second_centre_frequency = 0;
    second_bandwidth = 0;
    freq1_channels = get_nchan();
    freq2_channels = 0;
  }
  else{
    set_nchan( nchan1 + nchan2 );

    freq1_channels = nchan1;
    freq2_channels = nchan2;
    
    set_centre_frequency( freq_chan1_1 - 0.5 * chanbw1 + nchan1/2 * chanbw1 );
    second_centre_frequency = freq_chan1_2 - 0.5 * chanbw2 + nchan2/2 * chanbw2;

    set_bandwidth (chanbw1 * nchan1); 
    second_bandwidth = chanbw2 * nchan2; 
  }

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
  // OFFSET (49152=48*1024 samps per block)
  //
  
  uint64 offset_blocks = 0;

  string to_scan(&header[24],8);

  if( sscanf(to_scan.c_str(),UI64,&offset_blocks)!=1 )
    throw Error(FailedCall,"dsp::PMDAQ_Observation::PMDAQ_Observation()",
		"Failed to read in offset blocks");
  
  offset_blocks--; // Counting starts from one
  uint64 offset_bytes = 49152*offset_blocks;

  set_start_time (recording_start_time + get_nsamples(offset_bytes)/get_rate());

  /*
  static unsigned count = 0;
  count++;

  fprintf(stderr,"Got offset="UI64" Got start=%s so diff=%f\n",
	  offset_blocks,get_start_time().printall(),
	  get_nsamples(offset_bytes)/get_rate());
  if( count==10 )
    exit(0);
  */

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




