/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/WAPPFile.h"
#include "Error.h"

// from sigproc-2.4
#include "key.h"
// from sigproc-2.4

#define KEVINS_CODE
#ifdef KEVINS_CODE
#include "wapp_head.h"
#else
#include "wapp_header.h"
#endif

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" double wappcorrect (double mjd);

using namespace std;

dsp::WAPPFile::WAPPFile (const char* filename) : File ("WAPP")
{
  header = 0;
  if (filename)
    open_file (filename);
}

dsp::WAPPFile::~WAPPFile ( )
{
  if (header)
    free (header);
}


bool dsp::WAPPFile::is_valid (const char* filename) const
{
#ifdef KEVINS_CODE
  struct WAPP_HEADER header;
  int fd;
  fd=::open(filename,O_RDONLY);
  if(fd==-1)
    return false;
  try
  {
    if (verbose)
      cerr << "dsp::WAPPFile::is_valid try readheader" << endl;
    readheader(fd,&header);
  }
  catch(Error& error)
  {
    if (verbose)
      std::cerr << "dsp::WAPPFile::is_valid " << error.get_message() << endl;
    ::close(fd);
    return false;
  }  
  ::close(fd);
#else
  struct HEADERP* h = head_parse( filename );

  if (!h)
    return false;

  close_parse( h );
#endif
  return true;
}

#define fetch(N) fetch_hdrval(h,#N,&(head->N),sizeof(head->N))

void dsp::WAPPFile::open_file (const char* filename)
{
  header = malloc(sizeof(struct WAPP_HEADER));

  struct WAPP_HEADER* head = (struct WAPP_HEADER*) header;

#ifdef KEVINS_CODE
  fd=::open(filename,O_RDONLY);
  if(fd==-1)
    throw Error(FailedSys,"DSP::WAPPFile::Open_File","Could not open %s",filename);
  readheader(fd,head);
#else
  struct HEADERP* h = head_parse( filename );

  if (!h)
    throw Error (InvalidParam, "dsp::WAPPFile::open_file",
		 "not a WAPP file");


  fetch(src_name);
  fetch(obs_type);

  /* user-requested length of this integration (s) */
  fetch(obs_time);
  /* size (in bytes) of this header (nom =1024) */
  fetch(header_size);

  fetch(obs_date);
  fetch(start_time);

  /* user-requested sample time (us) */
  fetch(samp_time);
  fetch(wapp_time);

  fetch(num_lags);

  fetch(nifs);

  /* user-requested: 1 means 3-level; 2 mean 9-level  */
  fetch(level);

  fetch(lagformat);

  /* if we truncate data (0 no trunc)                 */
  /* for 16 bit lagmux modes, selects which 16 bits   */
  /* of the 32 are included as data                   */
  /* 0 is bits 15-0 1,16-1 2,17-2...7,22-7            */
  fetch(lagtrunc);

  fetch(cent_freq);

  fetch(bandwidth);

  fetch(freqinversion);

  /* requested ra J2000 (10000*hr+100*min+sec) */
  fetch(src_ra);

  /* requested dec J2000 (10000*deg+100*min+sec) */
  fetch(src_dec);

  fetch(start_az);
  fetch(start_za);
  fetch(start_ast);
  fetch(start_lst);

  /* user-requested: 1 means that data is sum of IFs  */
  fetch(sum);

  fetch(project_id);
  fetch(observers);

  fetch(psr_dm);

  fetch(dumptime);

  /* get number of bins which will be non zero in folding mode */
  fetch(nbins);

  // close_parse(h);
  fd = h->fd;
#endif

  cerr << "LEVEL=" << head->level << endl;

  // ////////////////////////////////////////////////////////////////////
  //
  // mode
  //
  /* what kind of observation is this */
  get_info()->set_mode(head->obs_type);

  // ////////////////////////////////////////////////////////////////////
  //
  // source
  //
  /* user-supplied source name (usually pulsar name) */
  get_info()->set_source (head->src_name);

  cerr << "Source = " << get_info()->get_source() << endl;

  // ////////////////////////////////////////////////////////////////////
  //
  // centre_frequency
  //
  /* user-supplied band center frequency (MHz) */
  get_info()->set_centre_frequency (head->cent_freq);

  // ////////////////////////////////////////////////////////////////////
  //
  // bandwidth
  //
  /* total bandwidth (MHz) for this observation */
  double bandwidth = head->bandwidth;
  /* 1 band is inverted, else band is not inverted    */
  if (head->freqinversion)
    bandwidth = -bandwidth;
  get_info()->set_bandwidth (bandwidth);

  // ////////////////////////////////////////////////////////////////////
  //
  // npol
  //
  /* user-requested: number of IFs to be recorded     */
  get_info()->set_npol (head->nifs);

  // ////////////////////////////////////////////////////////////////////
  //
  // state
  //
  if (head->nifs == 4)
    get_info()->set_state (Signal::Coherence);
  else if (head->nifs == 2)
    get_info()->set_state (Signal::PPQQ);
  else
    get_info()->set_state (Signal::Intensity);

  // ////////////////////////////////////////////////////////////////////
  //
  // nchan
  //
  /* user-requested number of lags per dump per spect */
  get_info()->set_nchan (head->num_lags);

  // ////////////////////////////////////////////////////////////////////
  //
  // nbit
  //
  /* 0=16 bit uint lags , 1=32 bit uint lags          */
  /* 2=32 bit float lags, 3=32 bit float spectra      */
  switch (head->lagformat) {
  case 0:
    get_info()->set_nbit (16);
    break;
  case 1:
    get_info()->set_nbit (32);
    break;
  case 3: /* timing mode data - not relevant, but needs to work! */
    break;
  case 4:
    get_info()->set_nbit (8);
    break;
  default:
    throw Error (InvalidState, "dsp::WAPPFile::open_file",
		 "lagformat variable in header should be 0, 1 or 4");
    break;
  }

  cerr << "WAPP nbit=" << get_info()->get_nbit() << endl;

  // ////////////////////////////////////////////////////////////////////
  //
  // start_time
  //

  /* built by WAPP from yyyymmdd */  
  string utc = head->obs_date;  utc += "-";

  /* UT seconds after midnight (start on 1-sec tick) [hh:mm:ss] */
  utc += head->start_time;

#if WVS_FIXES_STR2TM

  struct tm time;

  /* the str2tm function has been deprecated in favour of the standard C strptime */
  if (str2tm (&time, utc.c_str()) < 0)
    throw Error (InvalidState, "dsp::WAPPFile::open_file",
		 "Could not parse UTC from " + utc);

  MJD mjd (time);

#else

  // copied from WAPPArchive

  struct tm obs_date_greg;
  struct WAPP_HEADER* hdr = head;

  int rv = sscanf(hdr->obs_date, "%4d%2d%2d", 
      &obs_date_greg.tm_year, &obs_date_greg.tm_mon,
      &obs_date_greg.tm_mday);
  obs_date_greg.tm_year -= 1900;
  obs_date_greg.tm_mon -= 1;
  if (rv!=3) 
    throw Error (InvalidState, "dsp::WAPPFile::open_file",
        "Error converting obs_date string (rv=%d, obs_date=%s)", 
        rv, hdr->obs_date);
  rv = sscanf(hdr->start_time, "%2d:%2d:%2d", 
      &obs_date_greg.tm_hour, &obs_date_greg.tm_min, 
      &obs_date_greg.tm_sec);
  if (rv!=3) 
    throw Error (InvalidState, "dsp::WAPPFile::open_file",
        "Error converting start_time string (rv=%d, start_time=%s)", 
        rv, hdr->start_time);

  MJD mjd (obs_date_greg); 

#endif

  char buff[64];
  cerr << "UTC=" << utc << " MJD=" << mjd << " -> "
       << mjd.datestr (buff, 64, "%Y-%m-%d %H:%M:%S") << endl;

  // from sigproc-2.4
  /* for data between April 17 and May 8 inclusive, the start times are
     off by 0.5 days due to the ntp daemon not running... fix here.
     this also occured on May 17! hopefully will not happen again... */
  if ( ((mjd.intday() >= 52016) && (mjd.intday() <= 52039)) 
       || (mjd.intday() == 52046.0)) {
    cerr << "WARNING: MJD start time off by 0.5 days! fixed..." << endl;
    MJD half_day (0.5);
    mjd -= half_day;
  }

  get_info()->set_start_time (mjd);

  // ////////////////////////////////////////////////////////////////////
  //
  // rate
  //
  /* actual sample time (us) i.e. requested+dead time */
  double tsamp_us = head->wapp_time;

  // from sigproc-2.4
  tsamp_us += wappcorrect( mjd.in_days() );

  get_info()->set_rate ( 1e6 / tsamp_us );

  // ////////////////////////////////////////////////////////////////////
  //
  // telscope code
  //
  get_info()->set_telescope ("Arecibo");  // assume Arecibo

  string prefix="wapp";
  get_info()->set_machine("WAPP");	

  header_bytes = lseek(fd,0,SEEK_CUR);

  cerr << "header bytes=" << header_bytes << endl;

  set_total_samples();

}

