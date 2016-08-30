/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/S2File.h"
#include "Error.h"

// S2 header and unpacking routines
#include "tci_file.h"

#include <stdlib.h>
#include <string.h>

using namespace std;

// #define _DEBUG

dsp::S2File::S2File (const char* filename)
  : File ("S2")
{
  if (filename)
    open (filename);
}


bool dsp::S2File::is_valid (const char* filename) const
{ 
  tci_fd   s2file;
  tci_hdr  header;

  if (verbose)
    tci_file_verbose = 1;

  if (tci_file_open (filename, &s2file, &header, 'r') != 0)
    return false;

  return true;
}

/*! 
  Loads the Observation information from an S2-TCI style file.
*/
void dsp::S2File::open_file (const char* filename)
{
  load_S2info(filename);

  tci_fd   s2file;
  tci_hdr  header;

  if (tci_file_open (filename, &s2file, &header, 'r') != 0)
    throw Error (FailedCall, "dsp::S2File::open",
		 "tci_file_open (%s)", filename);

  fd = s2file.fd;

  for (int c=0; c<TCI_TIME_STRLEN-1; c++)
    if (!isdigit(header.hdr_time[c]))
	throw Error (InvalidState, "dsp::S2File::open",
		     "corrupted time in header");
    
  get_info()->set_identifier ("s" + string (header.hdr_time));
    
  utc_t utc;
  str2utc (&utc, header.hdr_time);  
  get_info()->set_start_time (utc);
    
  if (verbose) {
    char buffer [50];
    cerr << "dsp::S2File::open source_start_time"
	"\n  " << header.hdr_time << 
	"\n  " << utc2str(buffer, utc, "yyyy-ddd-hh:mm:ss") <<
	"\n  " << get_info()->get_start_time().printall() << endl;
  }
    
  get_info()->set_mode (header.hdr_s2mode);
  
  // find the '-' in the mode (assumed to be of format like 8x16-2)
  char* bitspersample = strrchr (header.hdr_s2mode, '-');
  
  if (!bitspersample) {
    cerr << "dsp::S2File::open - trouble finding bits/sample in " 
	   << header.hdr_s2mode
	   << "\ndsp::S2File::open -  setting to 2 bit/sample" << endl;
    get_info()->set_nbit (2);
  }
  else
    get_info()->set_nbit (atoi (bitspersample+1));
    
  get_info()->set_npol (2);
    
  if ( extra_hdr.source.length() > 1)
    get_info()->set_source (extra_hdr.source);
    
  else if (strlen(header.hdr_usr_field2) < 8)
    cerr << "dsp::S2File::open Warning: TCI header field2 ("
	   << header.hdr_usr_field2 << ") lacks source" << endl;
    
  else
    get_info()->set_source (header.hdr_usr_field2);
    
  double centre_frequency = 0.0;
    
  if (extra_hdr.freq != 0.0){
    centre_frequency = extra_hdr.freq;
  }
  else if (sscanf (header.hdr_usr_field3, "%lf", &(centre_frequency)) != 1) {
    cerr << "dsp::S2File::open Warning: TCI header field3 ("
	   << header.hdr_usr_field3 << ") lacks frequency" << endl;
    centre_frequency = 0.0;
  }
    
  get_info()->set_centre_frequency (centre_frequency);
    
  // S2 data defaults to single side-band, real-sampled data
  get_info()->set_state (Signal::Nyquist);
    
  // tci_file_open returns data_rate in W/s (16bit/s)
  get_info()->set_rate (double (s2file.data_rate) * 2.0 / get_info()->get_nbyte());
  get_info()->set_bandwidth (16.0);
  
  get_info()->set_machine ("S2");
  if(extra_hdr.telid > ' ' )
    get_info()->set_telescope ( string (1, extra_hdr.telid) );
  else
    get_info()->set_telescope ( "PKS" );
    
  // tci_file_open returns file size in Words (16 bits)
  get_info()->set_ndat ( int64_t(s2file.fsz) * 16 / (get_info()->get_nbit()*get_info()->get_npol()) );
    
  if (verbose)
    cerr << "dsp::S2File::open " << s2file.fsz * 2 << " bytes = "
	   << get_info()->get_ndat() << " time samples" << endl;
    
  header_bytes = s2file.base;
  
  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / get_info()->get_nbit();
  if (resolution == 0)
    resolution = 1;

  if (verbose)
    cerr << "dsp::S2File::open return" << endl;
}

void dsp::S2File::load_S2info (const char *filename)
{
  extra_hdr.source = "";
  extra_hdr.telid = 0;
  extra_hdr.freq = 0.0;
  extra_hdr.calperiod = 0.0;
  extra_hdr.tapeid = "";

  FILE *S2Info;
  static char* linebuf= new char[40];
  char* whitespace = "\n\t ";
    
  string *info_file = new string(filename);
  
  // replace the ".psr" in the filename with ".info" for the INFO file
  int ext_pos = info_file->find_last_of(".");
  info_file->erase(ext_pos+1, 3);
  info_file->append("info");
  
  if (verbose)
    cerr << "Will attempt to open " << *info_file << " as S2 Extra Information file" << endl;
  
  S2Info = fopen(info_file->c_str(), "r");
  
  if(S2Info == NULL)
    throw Error (FailedCall, "dsp::S2File::load_S2info",
		 "no extra info file found (%s)", filename);
  
  while(fgets(linebuf, 40, S2Info) != NULL) {
    char* key   = strtok (linebuf, whitespace);
    char* value = NULL;
    if (!key)
      continue;
    value = strtok (NULL, whitespace);
    if (!value)
      continue;
    
    if (strcmp (key, "SOURCE") == 0) {
      extra_hdr.source = value;
    }
    else if (strcmp (key, "TELID") == 0) {
      if (sscanf (value, "%c", &extra_hdr.telid) < 1)  {
	perror ("obshandle::load_S2info: error parsing TELESCOPE CODE");
	break;
      }
    }
    else if (strcmp (key, "FREQ") == 0) {
      if (sscanf (value, "%lf", &extra_hdr.freq) < 1) {
	perror ("obshandle::load_S2info: error parsing FREQUENCY");
	break;
      }
    }
    else if (strcmp (key, "CALPERIOD") == 0) {
      if (sscanf (value, "%lf", &extra_hdr.calperiod) < 1) {
	perror ("obshandle::load_S2info: error parsing CALPERIOD");
	break;
      }
    }
    else if (strcmp (key, "TAPEID") == 0) {
      extra_hdr.tapeid = value;
    }
  } /* while */

  fclose (S2Info);

  if(verbose){
    cerr << "Source: " << extra_hdr.source << endl;
    cerr << "Telid : " << (char)extra_hdr.telid  << endl;
    cerr << "Freq  : " << extra_hdr.freq   << endl;
    cerr << "Tapeid: " << extra_hdr.tapeid << endl;
  }
}

