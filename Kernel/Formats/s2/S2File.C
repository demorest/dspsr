//#include <fcntl.h>
//#include <stdio.h>
//#include <assert.h>
//#include <unistd.h>

// S2 header and unpacking routines
#include "S2File.h"
#include "Telescope.h"

#include "tci_file.h"
#include "genutil.h"

bool dsp::S2File::is_valid (const char* filename) const
{ 
  tci_fd   s2file;
  tci_hdr  header;

  if (tci_file_open (filename, &s2file, &header, 'r') != 0)
    return false;

  return true;
}

/*! 
  Loads the Observation information from an S2-TCI style file.
*/
void dsp::S2File::open (const char* filename)
{
  tci_fd   s2file;
  tci_hdr  header;

  if (tci_file_open (filename, &s2file, &header, 'r') != 0)
    throw_str ("S2File::open - could not construct from %s\n", filename);

  for (int c=0; c<TCI_TIME_STRLEN-1; c++)
    if (!isdigit(header.hdr_time[c]))
      throw_str ("S2File::open - corrupted time in header.\n");

  info.set_identifier ("s" + string (header.hdr_time));

  utc_t utc;
  str2utc (&utc, header.hdr_time);  
  info.set_start_time (utc);

#ifdef _DEBUG
  char buffer [50];
  fprintf (stderr, "S2File::open source_start_time: %s->%s->%s \n",
	   header.hdr_time, utc2str(buffer, utc, "yyyy-ddd-hh:mm:ss"),
	   info.get_start_time.printall());
#endif
  
  info.set_mode (header.hdr_s2mode);
  
  // find the '-' in the mode (assumed to be of format like 8x16-2)
  char* bitspersample = strrchr (header.hdr_s2mode, '-');

  if (!bitspersample) {
    cerr << "S2File::open - trouble finding bits/sample in " 
	 << header.hdr_s2mode
	 << "\nS2File::open -  setting to 2 bit/sample" << endl;
    info.set_nbit (2);
  }
  else
    info.set_nbit (atoi (bitspersample+1));

  info.set_npol (2);
  
  if (strlen(header.hdr_usr_field2) < 8)
    cerr << "S2File::open Warning: TCI header field2 ("
	 << header.hdr_usr_field2 << ") lacks source" << endl;
	 
  else
    info.set_source (header.hdr_usr_field2);

  double centre_frequency = 0.0;

  if (sscanf (header.hdr_usr_field3, "%lf", &(centre_frequency)) != 1) {
    cerr << "S2File::open Warning: TCI header field3 ("
	 << header.hdr_usr_field3 << ") lacks frequency" << endl;
    centre_frequency = 0.0;
  }
  
  info.set_centre_frequency (centre_frequency);
  
  // S2 data defaults to single side-band, real-sampled data
  info.set_state (Signal::Nyquist);

  // tci_file_open returns data_rate in W/s (16bit/s)
  info.set_rate (double (s2file.data_rate) * 2.0 / info.nbyte());
  info.set_bandwidth (16.0);
  
  fd = s2file.fd;
  header_bytes = s2file.base;
  

#if 0  
  // tci_file_open returns file size in Words (16 bits)
  info.set_total_data = (Int64) s2file.fsz * 16.0 / (info.set_nbit*info.set_npol);
  
#ifdef DEBUG_BITSTREAM
  fprintf (stderr, "S2File::open - data %ld bytes %ld time samp\n",
	   s2file.fsz * 2, info.set_total_data);
  fprintf (stderr, "S2File::open - header size %d\n",
	   info.set_header_size);
  
  fflush (stderr);
#endif
  
#endif

  info.set_machine ("S2");

  info.set_telescope (Telescope::Parkes);

  info.set_default_basis();

  // set the file pointers
  reset();

  if (verbose)
    cerr << "Returning from S2File::open" << endl;
}

