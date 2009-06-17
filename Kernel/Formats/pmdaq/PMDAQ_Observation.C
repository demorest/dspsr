/***************************************************************************
 *
 *   Copyright (C) 2002 by Matthew Bailes
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PMDAQ_Observation.h"
#include "strutil.h"
#include "coord.h"

using namespace std;

// See /psr/cvshome/hknight/soft_swin/search/sc_td/sc_pmhead.inc for more details on header format

dsp::PMDAQ_Observation::PMDAQ_Observation(const char* header) : Observation()
{
  if (header == NULL)
    throw Error(InvalidState,"PMDAQ_Observation::PMDAQ_Observation",
		"no header passed in!");

  int jump = 0;
  if( header[0] == 'P' )
    jump = -4;

  // //////////////////////////////////////////////////////////////////////
  //
  // PMDAQ_HEADER_VERSION
  //
  float version;

  int nscanned;

  nscanned = sscanf(&header[10+jump],"%f4.1",&version);
  if (nscanned != 1) {
    throw Error (InvalidState, "PMDAQ_Observation", "failed read PMDAQ_HEADER_VERSION");
  }

  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE
  //

  char telescope_name[13];
  sscanf(&header[592+jump],"%s",telescope_name);
  set_telescope (telescope_name);

  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  char pmdaq_source_name[13];
  sscanf(&header[528+jump],"%s",pmdaq_source_name);

  set_source (pmdaq_source_name);

  // //////////////////////////////////////////////////////////////////////
  //
  // nchan(s), centre_frequency(s), bandwidth(s) 
  //
  unsigned nfilters = read_header<unsigned>(header,206+jump,2);

  double chanbw1 = read_header<double>(header,208+jump,8);
  double chanbw2 = read_header<double>(header,216+jump,8);

  unsigned nchan1 = read_header<unsigned>(header,224+jump,4);
  unsigned nchan2 = read_header<unsigned>(header,228+jump,4);

  double freq_chan1_1 = read_header<double>(header,232+jump,12);
  double freq_chan1_2 = read_header<double>(header,244+jump,12);

  if( nfilters==1 ){
    set_nchan( nchan1 );

    if( get_nchan()==512 && fabs(chanbw1) > 2.9 ){
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

    fprintf(stderr,"Got 2 filters for pmdaq file with nchan1=%d nchan2=%d cf1=%f cf2=%f bw1=%f bw2=%f\n",
	    nchan1, nchan2,
	    freq_chan1_1 - 0.5 * chanbw1 + nchan1/2 * chanbw1, freq_chan1_2 - 0.5 * chanbw2 + nchan2/2 * chanbw2,
	    chanbw1 * nchan1, chanbw2 * nchan2);
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

  // //////////////////////////////////////////////////////////////////////
  //
  // TSAMP
  //
  double sampling_interval;

  nscanned = sscanf(&header[jump+256],"%lf",&sampling_interval);
  if (nscanned != 1) {
    throw Error (InvalidState, "PMDAQ_Observation", "failed to read sampling interval\n");
  }

  sampling_interval *= 1e-3;

  cerr << " Sampling interval is " << sampling_interval << " secs " << endl;

  set_rate (1.0/sampling_interval);

  // //////////////////////////////////////////////////////////////////////
  //
  // MJD_START
  //

  // MXB MJD is at header[jump+40], UT is at header[jump+48]

  int int_MJD;
  int hh,mm;
  double secs;

  // The approximately one millionth UT to secs parser.

  nscanned = sscanf(&header[jump+40],"%d",&int_MJD);
  if (nscanned != 1) {
    throw Error (InvalidState, "PMDAQ_Observation", "failed to read integer part of MJD\n");
  }
  nscanned = sscanf(&header[jump+49],"%2d",&hh);
  if (nscanned != 1) {
    throw Error (InvalidState, "PMDAQ_Observation", "failed to read hours in ut start\n");
  }
  nscanned = sscanf(&header[jump+52],"%2d",&mm);
  if (nscanned != 1) {
    throw Error (InvalidState, "PMDAQ_Observation", "failed to read minutes in ut start\n%2.2s\n",&header[jump+51]);
  }
  nscanned = sscanf(&header[jump+55],"%9lf",&secs);
  if (nscanned != 1) {
    throw Error (InvalidState, "PMDAQ_Observation", "failed to read seconds in ut start\n");
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
  
  uint64_t offset_blocks = 0;

  string to_scan(&header[jump+24],8);

  if( sscanf(to_scan.c_str(),UI64,&offset_blocks)!=1 )
    throw Error(FailedCall,"dsp::PMDAQ_Observation::PMDAQ_Observation()",
		"Failed to read in offset blocks");
  
  offset_blocks--; // Counting starts from one
  uint64_t offset_bytes = 49152*offset_blocks;

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
  // i.e. centre frequency is true centre frequency
  //
  dc_centred = false;

  // make an identifier name
  set_mode (stringprintf ("%d-bit mode", get_nbit()));
  set_machine ("PMDAQ");

  // //////////////////////////////////////////////////////////////////////
  //
  // RA and DEC
  //
  double ra, dec;
  bool has_position;

  has_position = (str2ra  (&ra,  &header[jump+78]) == 0);
  has_position = (str2dec2 (&dec, &header[jump+94]) == 0);

  if (!has_position) {
    ra = dec = 0.0;
  }

  coordinates.setRadians(ra, dec);
}




