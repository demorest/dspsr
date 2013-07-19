/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/FadcObservation.h"

#include "fadc_header.h" // needed to read header and block.map
#include "coord.h"

#include <fstream>

using namespace std;

dsp::FadcObservation::FadcObservation (const char* header)
{
}


int dsp::FadcObservation::read_blockMap( long *buffers_per_file, long *bytes_per_buffer)
{
  // read block.map find bytes per buffer and buffers per file
  std::ifstream block_map ("block.map");
  if (!block_map)
  {
      // default: buffers_per_file = number of digital channels from Exp file
    *bytes_per_buffer = 1048576;
    cerr<<"FadcFile: Cannot open block.map, assume "<<(*buffers_per_file)<<" buffers per file, 1048576 bytes per buffer.";
    return -1;
  }
  else 
  {
    const char* line="";
    char ch;
    string str="";
    
    while (block_map.get(ch)) str+=ch; 
    
    line = str.c_str();
    
    if (fadc_blockmap_get(line, "Each File is", "%ld", buffers_per_file) < 0)
    {
      cerr<<"FadcFile - failed read buffers_per_file from block.map";
      // default: buffers_per_file number of digital channels from Exp file
    }
 
    if (fadc_blockmap_get(line, "Each Buffer is", "%ld", bytes_per_buffer) < 0)
    {
      cerr<<"FadcFile - failed read bytes_per_buffer from block.map, assume 1048576";
      *bytes_per_buffer = 1048576;
    }
    
    block_map.close();
cerr<<"\nRead block.map\n   buffers_per_file = "<<*buffers_per_file<<"\n   bytes_per_buffer = "<<*bytes_per_buffer<<"\n\n";
  }  // block.map taken care of
  return 0;
}


dsp::FadcObservation::FadcObservation (const char* header, long firstFile, long lastFile, unsigned long offset_tsmps_file0, unsigned long offset_tsmps, double centerFreqOverride, double bwOverride)
{
  if (header == NULL)
    throw Error (InvalidState, "FadcObservation", "no header!");

  // //////////////////////////////////////////////////////////////////////
  //
  // TELESCOPE - Arecibo, GBT, VLA  (VLA not implemented yet)
  //
  char hdrstr[64];
  if (fadc_header_get (header, "Observatory Name", "%s", hdrstr) < 0)
    throw Error (InvalidState, "FadcObservation", "failed read Observatory Name");
  
  string tel = hdrstr;
  if ( tel == "Arecibo" || tel == "AO" || tel=="ARECIBO" || tel=="arecibo" || tel=="Ao" || tel=="ao") 
    get_info()->set_telescope ("3");
  else if ( tel == "Greenbank" || tel == "GBT" || tel=="GreenBank" || tel=="greenbank" || tel=="Gbt" || tel=="gbt" || tel=="GREENBANK" ) 
    get_info()->set_telescope ("1");
  else {
    cerr << "FadcObservation:: Warning using telescope code " << hdrstr << endl;
    get_info()->set_telescope (hdrstr);
  }
cerr<<"FadcObservation: Telescope = "<<tel<<"\n";
  // //////////////////////////////////////////////////////////////////////
  //
  // SOURCE
  //
  if (fadc_header_get (header, "Source Name", "%s", hdrstr) < 0)
    throw Error (InvalidState, "FadcObservation", "failed read SOURCE");

  set_source (hdrstr);
cerr<<"FadcObservation: Source = "<<hdrstr<<"\n";

  // //////////////////////////////////////////////////////////////////////
  //
  // FREQ  in MHz
  //       This assumes that LCP and RCP have the same center freq (look at "Channel 0")
  double freq;
  if (fadc_header_get (header, "Center Freq (Hz)", "%lf", &freq) < 0)
    throw Error (InvalidState, "FadcObservation", "failed read FREQ");

  cerr<<"Center Frequency according to Exp-File = "<<(freq/1000000.)<<" MHz\n";
  if (centerFreqOverride != 0)
  {
     cerr<<"Override center frequency = "<<(centerFreqOverride/1000000.)<<" MHz\n";
     freq=centerFreqOverride;
  }  
  set_centre_frequency (freq/1000000.);
cerr<<"FadcObservation: Freq MHz = "<<(freq/1000000.)<<"\n";

  // //////////////////////////////////////////////////////////////////////
  //
  // BW in MHz
  //
  double bw, expBW, expSmpRate;
  if (fadc_header_get (header, "IF Bandwidth (Hz)", "%lf", &expBW) < 0)
    throw Error (InvalidState, "FadcObservation", "failed read BW");
  if (fadc_header_get (header, "Sample Rate (Hz)", "%lf", &expSmpRate) < 0)
    throw Error (InvalidState, "FadcObservation", "failed read Sampling Rate");

  cerr<<"Bandwidth according to Exp-File = "<<(expBW/1000000.)<<" MHz\n";
  cerr<<"Sampling rate according to Exp-File = "<<(expSmpRate/1000000.)<<" MHz (more likely to be correct)\n";
  if (bwOverride != 0)
  {
     cerr<<"Override bandwidth = "<<(bwOverride/1000000.)<<" MHz\n";
     bw=bwOverride;
  }
  else bw=expSmpRate;
    
  set_bandwidth (bw/1000000.);
cerr<<"FadcObservation: BW = "<<(bw/1000000.)<<"\n";

  //
  // FADC data is usually one channel, 2 pol (LCP, RCP) each with a real and a complex stream
  //    Generalize this later
  set_nchan(1);

  // //////////////////////////////////////////////////////////////////////
  //
  // NPOL 
  //  
  int nADCperCard;
  if (fadc_header_get (header, "Number of ADC", "%d", &nADCperCard) < 0)
  throw Error (InvalidState, "FadcObservation", "failed read number of ADC per Card");

  int nChan=0;
  // The number of Cards with ADCs is the same as the number of digital channels
  // count "NumDigital_Channel No" in header
  {
    const char* tmp = header;
    do 
    {
      tmp=strstr (tmp, "Digital_Channel No");
      if (tmp) 
      {
         nChan++;
	 tmp++;
      }
    } while (tmp);
  }
cerr<<"FadcObservation: Found "<<nChan<<" digital channels\n";

  if ((nChan != 1)&&(nChan != 2)&&(nChan != 4)) 
  {
    cerr<<"Found "<<nChan<<" digital channels\n";
    throw Error (InvalidState, "FadcObservation", "Unexpected Format: Expected 1, 2 or 4 digital channels");
  }
  
  int scan_npol=0;
  if ((nADCperCard==1)&&(nChan==4)) scan_npol=2;    // standard case since August 2003
  else if ((nADCperCard==2)&&(nChan==2)) scan_npol=2; // standard before August 2003, at that time no magic codes
  else if ((nADCperCard==1)&&(nChan==2)) scan_npol=1;
  else if ((nADCperCard==2)&&(nChan==1)) scan_npol=1;
  else if ((nADCperCard==2)&&(nChan==4))
  {
    scan_npol=2;
    throw Error (InvalidState, "FadcObservation", "Unexpected Format: 2pol, 2frequencies(?) has not been implemented yet. [4 digital channels with 2 ADC per card]");
  }
  else
  {
     cerr<<"FadcObservation: Unknown data format:   nADCperCard = "<<nADCperCard<<"     nChan = "<<nChan<<"\n";
     throw Error (InvalidState, "Fatal Error (Unknown data format)");
  }
cerr<<"FadcObservation: Found "<<scan_npol<<" polarizations\n";

  set_npol (scan_npol);

  // //////////////////////////////////////////////////////////////////////
  //
  // NBIT - Assume that all NBIT is the same for RCP and LCP
  //        NBIT means bits per value (ie LCP, real)
  int scan_nbit;
  if (fadc_header_get (header, "Sample Res. (Bits)", "%d", &scan_nbit) < 0)
    throw Error (InvalidState, "FadcObservation", "failed read NBIT");

  set_nbit (scan_nbit);
cerr<<"FadcObservation: Sample Res. Bits = "<<scan_nbit<<"\n";

  // //////////////////////////////////////////////////////////////////////
  //
  // NDIM -  FADC data is quadrature sampled
  //
  set_state (Signal::Analytic);  // Defined in Types.h
                                 // Analytic = "In-phase and Quadrature 
				 //          sampled voltages (complex)"

  //
  // call this only after setting frequency and telescope
  //

  // //////////////////////////////////////////////////////////////////////
  //
  // Sampling Rate
  // samples per second (Hz)
  double sampling_freq;
  if (fadc_header_get (header, "Sample Rate (Hz)", "%lf", &sampling_freq)<0)
    throw Error (InvalidState, "FadcObservation", "failed read sampling frequency");

  set_rate (sampling_freq);
cerr<<"FadcObservation: Sampling Freq Hz = "<<sampling_freq<<"\n";

  // //////////////////////////////////////////////////////////////////////
  //
  // MJD_START   
  // Unix time = sec since start of 1970        MJD = days since Nov 17 1858 0:00
  uint64_t unix_start;
  if (fadc_header_get (header, "Unix Start Time", UI64, &unix_start) < 0)
    throw Error (InvalidState, "FadcObservation", "failed read Unix Start Time");
  
  //                      days since 1970   +   days between Nov 17th 1858 and 1/1/1970 0:00
  //                                            45 days in 1858, 111years, 27 times Feb 29th (every four years except 1900)
  double mjd_start = unix_start / (86400.)  +   45 + 365*111 + 27;

  // according to MJD.h the double MJD constructor expects days
  MJD recording_start_time (mjd_start);
cerr<<"FadcObservation: MJD Start (Days since Nov 17 1858) = "<<mjd_start<<"\n";

  // //////////////////////////////////////////////////////////////////////
  //
  // OFFSET
  //
  
  long buffers_per_file=nChan;
  long bytes_per_buffer=0;
  read_blockMap(&buffers_per_file, &bytes_per_buffer);
  
  uint64_t samples_per_file = (uint64_t) buffers_per_file * bytes_per_buffer *4 / (scan_nbit * scan_npol); 
         // each sample consists of two nbit numbers per polarization
        // so each sample has 2*nbit*npol bit, 
        // nbit*npol/4  * number of samples=number of bytes
  
  uint64_t offset_samples = samples_per_file * firstFile;  // because files are numbered starting with zero
cerr<<"FadcObservation: offset samples due to choice of first file = "<<offset_samples<<"\n";

// NOTE: expect magic code since mjd_start>52852.  (CHECK THIS !!)  1.August 2003 (or some other time in Aug. 2003)

  if ((offset_tsmps_file0!=0)||(offset_tsmps!=0))   // magic code case
  {
    offset_samples += -(offset_tsmps_file0 - 24*2/(scan_nbit * nADCperCard)) + offset_tsmps;
                                              // starting this many samples from start of original firstFile
                 // time starts counting 24/12/6samples [half as many if re an im in the same digiChannel] (6bytes) before data in original file 0 (start of magic code)
                 // we have counted the first samples of the file for the offset although the clock didn't start
                 // running yet, so we have to substract these samples again
cerr<<"FadcObservation: offset samples due to shifts and Magic Code = "<<-(offset_tsmps_file0 - 24*2/(scan_nbit * nADCperCard)) + offset_tsmps<<"\n";
  }
  
  double sampling_interval = 1.0/sampling_freq;
  
  double offset_seconds = double(offset_samples) * sampling_interval;
cerr<<"offset s = "<<offset_seconds<<"\n";
  set_start_time (recording_start_time + offset_seconds);  // The MJD + double operator expects double to be seconds
               
cerr<<"FadcObservation: Recording Start time set\n";  

  // //////////////////////////////////////////////////////////////////////
  //
  // Number of timesamples in measurement
  // set to zero if no idea
  // set_ndat( 0 );
  // 2 bit example: We have four 2bit values per timesample -> no timesamples = no bytes
  // note: we omitted offset_tsmps time samples in first file

  uint64_t total_samples = static_cast <uint64_t> ((lastFile-firstFile +1)) * buffers_per_file * bytes_per_buffer * 4 / (scan_npol * scan_nbit) - offset_tsmps;   
  set_ndat( total_samples );  // need -skipped samples
cerr<<"FadcObservation: used offset_tsmps = "<<offset_tsmps<<'\n';  
cerr<<"FadcObservation: total_samples = "<<total_samples<<'\n';
  //
  // until otherwise, the band is centred on the centre frequency
  //
  dc_centred = true;

  // //////////////////////////////////////////////////////////////////////
  //
  // PRIMARY - prefix: I think this is the leading character for names of produced files (archives)
  //
  string prefix = "F";

  // make an identifier name
  set_mode (tostring(get_nbit()) + "-bit mode");
  set_machine ("Fadc");
  
  // //////////////////////////////////////////////////////////////////////
  //
  // RA and DEC
  //
  bool has_position = true;
  double ra=0.0 , dec=0.0, ra_hrs=0.0, dec_deg=0.0;

  if (has_position){
    has_position = (fadc_header_get (header, "Right Ascension (Hr)", "%lf", &ra_hrs) >= 0);
    //    fprintf(stderr,"1 has_position=%d hdrstr='%s'\n",has_position,hdrstr);
  }

  if (has_position){
    ra = ra_hrs * 3.14159265358979 /12.;
  }

  if (has_position){
    has_position = (fadc_header_get (header, "Declination (Degrees)", "%lf", &dec_deg) >= 0);
    //fprintf(stderr,"3 has_position=%d hdrstr='%s'\n",has_position,hdrstr);
  }

  if (has_position){
    dec = dec_deg * 3.14159265358979 /180.;
  }

  if (!has_position){
    ra = dec = 0.0;
    //fprintf(stderr,"5 has_position=%d set shit to zero\n",has_position);
  }
  
  //fprintf(stderr,"Got ra=%f dec=%f exiting\n",ra,dec);
cerr<<"FadcObservation: ra  (hrs) = "<<ra_hrs <<"     ra  (rad) = "<<ra<<"\n";  
cerr<<"FadcObservation: dec (deg) = "<<dec_deg<<"     dec (rad) = "<<dec<<"\n";  

  coordinates.setRadians(ra, dec);
cerr<<"FadcObservation: radians set\n";  
cerr<<"FadcObservation: finished\n\n";  
}
