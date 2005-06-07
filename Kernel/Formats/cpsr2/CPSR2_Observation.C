#include "dsp/CPSR2_Observation.h"
#include "cpsr2_header.h"

#include "string_utils.h"
#include "genutil.h"
#include "Types.h"
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
  set_ndat( 0 );

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
  // MODE
  //
  if (ascii_header_get(header, "MODE", "%s", hdrstr) < 0)
    throw_str ("CPSR2_Observation - failed read MODE");

  if (strncmp(hdrstr,"PSR",3) == 0) {
    cerr << "Source is Pulsar" << endl;
    set_type(Signal::Pulsar);
  }
  else if (strncmp(hdrstr,"CAL",3) == 0) {
    cerr << "Source is CAL" << endl;
    set_type(Signal::PolnCal);
  }
  else if (strncmp(hdrstr,"LEV",3) == 0) {
    cerr << "Source is LEVCAL" << endl;
    set_type(Signal::PolnCal);
  }
  else {
    cerr << "Unknown mode or source type, assuming Pulsar" << endl;
    set_type(Signal::Pulsar);
  }
  if (get_type() == Signal::PolnCal) {

    double calfreq;
    if (ascii_header_get (header, "CALFREQ", "%lf", &calfreq) < 0)
      throw_str ("CPSR2_Observation - failed read FREQ");
    set_calfreq(calfreq);
    cerr << "Set CALFREQ to " << get_calfreq() << endl;

  }

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
  // CPSR2 data is single-channel unless this is a detected CPSR2 FB see later
  //
  set_nchan(1);

  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL
  //
  int scan_npol;
  if (ascii_header_get (header, "NPOL", "%d", &scan_npol) < 0)
    throw_str ("CPSR2_Observation - failed read NPOL");

  set_npol (scan_npol);

  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT
  //
  int scan_nbit;
  if (ascii_header_get (header, "NBIT", "%d", &scan_nbit) < 0)
    throw_str ("CPSR2_Observation - failed read NBIT");

  set_nbit (scan_nbit);

  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM
  //
  int scan_ndim;
  if (ascii_header_get (header, "NDIM", "%d", &scan_ndim) < 0)
    throw_str ("CPSR2_Observation - failed read NDIM");
  set_ndim(scan_ndim);

  switch (scan_ndim) {
  case 1:
    set_state (Signal::Nyquist); break;
  case 2:
    set_state (Signal::Analytic); break;
  default:
    throw_str ("CPSR2_Observation - invalid NDIM=%d\n", get_ndim());
  }

  // ////////////////////////////////////////////////////////////////////
  // TEST for detected SimpleFB filterbank file
  // ///////////////////////////////////////////////////////////////////

  int scan_nchan;
  if ((ascii_header_get (header,"NCHAN", "%d", &scan_nchan)) && (get_npol() == 1)){
	  set_state (Signal::Intensity);
	  set_nchan(scan_nchan);
	  set_thresh();
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
  uint64 bitspersample = get_nbit()*get_npol();

  uint64 offset_samples = offset_bytes * bitsperbyte / bitspersample;
  
  double offset_seconds = double(offset_samples) * sampling_interval;

  set_start_time (recording_start_time + offset_seconds);

  //
  // until otherwise, the band is centred on the centre frequency
  //
  dc_centred = false; // new
  //  dc_centred = true;  // old

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
  set_mode (stringprintf ("%d-bit mode", get_nbit()));
  set_machine ("CPSR2");

  // //////////////////////////////////////////////////////////////////////
  //
  // RA and DEC
  //
  bool has_position = true;
  double ra, dec;

  if (has_position){
    has_position = (ascii_header_get (header, "RA", "%s", hdrstr) == 1);
    //    fprintf(stderr,"1 has_position=%d hdrstr='%s'\n",has_position,hdrstr);
  }

  if (has_position){
    has_position = (str2ra (&ra, hdrstr) == 0);
    //fprintf(stderr,"2 has_position=%d ra=%f\n",has_position,ra);
  }

  if (has_position){
    has_position = (ascii_header_get (header, "DEC", "%s", hdrstr) == 1);
    //fprintf(stderr,"3 has_position=%d hdrstr='%s'\n",has_position,hdrstr);
  }

  if (has_position){
    has_position = (str2dec2 (&dec, hdrstr) == 0);
    //fprintf(stderr,"4 has_position=%d dec=%f\n",has_position,dec);
  }

  if (!has_position){
    ra = dec = 0.0;
    //fprintf(stderr,"5 has_position=%d set shit to zero\n",has_position);
  }
  
  //fprintf(stderr,"Got ra=%f dec=%f exiting\n",ra,dec);

  coordinates.setRadians(ra, dec);
}

void dsp::CPSR2_Observation::set_thresh() {
  thresh = new float [2*get_nchan()];
}    
