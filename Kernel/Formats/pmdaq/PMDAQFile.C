/***************************************************************************
 *
 *   Copyright (C) 2002 by Matthew Bailes
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PMDAQFile.h"
#include "dsp/PMDAQ_Observation.h"
#include "dsp/PMDAQ_Extension.h"
#include "dsp/BitSeries.h"
#include "dsp/Observation.h"

#include "dirutil.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>

using namespace std;

#define PMDAQ_HEADER_SIZE 648

bool dsp::PMDAQFile::using_second_band = false;

//! Destructor
dsp::PMDAQFile::~PMDAQFile(){}

dsp::PMDAQFile::PMDAQFile (const char* filename) 
  : BlockFile ("PMDAQ")
{
  if (filename)
    open (filename);

  block_header_bytes = 4;
  block_tailer_bytes = 4;
  block_bytes = 48*1024 + 4 + 4;

  chan_begin = -1;
  chan_end = -1;
}

int dsp::PMDAQFile::get_header (char* pmdaq_header, const char* filename)
{
  ////////////////////////////////////////////////////////
  // See if file has been dumped straight to disk using dd
  {
    int fd = ::open(filename,O_RDONLY);
    if( fd < 0 )
      throw Error(InvalidState,"dsp::PMDAQFile::get_header()",
		  "File '%s' does not even exist!",
		  filename);
    
    if( ::read( fd, pmdaq_header, 640) == 640 ){
      if( pmdaq_header[0] == 'P' && pmdaq_header[1] == 'M' && pmdaq_header[2] == 'D' ){
	block_header_bytes = 640;
	block_tailer_bytes = 0;
        block_bytes = 48*1024 + 640;
	return 0;
      }
    }
    ::close(fd);
  }

  ////////////////////////////////////////////////////////
  // See if sc_td has been used to get file onto disk
  string str_filename = string (filename);
  int pos = str_filename.find_last_of(".",str_filename.size()-1);
  string hdr_name = str_filename.substr(0,pos) + ".hdr";

  if( verbose )
    cerr << "dsp::PMDAQFile::get_header looks for header called " <<
      hdr_name << endl;

  int fd = ::open (hdr_name.c_str(), O_RDONLY);

  if (fd < 0) {
    if (verbose)
      fprintf (stderr, "PMDAQFile::get_header - failed open(%s): %s", 
	       hdr_name.c_str(), strerror(errno));
    return -1;
  }

  int retval = read (fd, pmdaq_header, PMDAQ_HEADER_SIZE);

  ::close (fd);    

  if (retval < PMDAQ_HEADER_SIZE) {
    if (verbose)
      fprintf (stderr, "PMDAQFile::get_header - failed read: %s",
	       strerror(errno));

    return -1;
  }
  return 0;
}

static char pmdaq_header [PMDAQ_HEADER_SIZE];

bool dsp::PMDAQFile::is_valid (const char* filename) const
{
  Reference::To<PMDAQFile> dummy = new PMDAQFile;

  if ( dummy->get_header (pmdaq_header, filename) < 0)
    return false;
    
  if (strncmp(&pmdaq_header[4],"PMD",3)!=0 && strncmp(&pmdaq_header[0],"PMD",3)!=0 )
    return false;

  if (verbose) printf("File %s is valid PMDAQ data!\n",filename);
  return true;
}

/*
  If user has requested channels that lie in the second observing band
  on disk, modify the bandwidth and centre frequency of output */

void dsp::PMDAQFile::modify_info(PMDAQ_Observation* data)
{
  if( verbose )
    fprintf(stderr,"In dsp::PMDAQFile::modify_info() with %d and %d cf=%f bw=%d\n",
	    data->has_two_filters(), using_second_band,
	    get_info()->get_centre_frequency(), get_info()->get_bandwidth());

  if( !data->has_two_filters() )
    return;
  
  if( !using_second_band ){
    set_chan_begin( 0 );
    set_chan_end( data->get_freq1_channels() );
    return;
  }

  set_chan_begin( data->get_freq1_channels() );
  set_chan_end( data->get_freq1_channels() + data->get_freq2_channels() );

  get_info()->set_centre_frequency( data->get_second_centre_frequency() );
  get_info()->set_bandwidth( data->get_second_bandwidth() );

  if( verbose )
    fprintf(stderr,"In dsp::PMDAQFile::modify_info() have set chan_begin to %d chan_end to %d cf=%f bw=%f\n",
	    chan_begin, chan_end, get_info()->get_centre_frequency(), get_info()->get_bandwidth());
}

void dsp::PMDAQFile::open_file (const char* filename)
{  
  if (get_header (pmdaq_header, filename) < 0)
    throw Error(InvalidState,"PMDAQFile::open",
		"failed get_header(%s): %s",
		filename, strerror(errno));
  
  PMDAQ_Observation* data = new PMDAQ_Observation (pmdaq_header);
  modify_info(data);
  info = data;

  // Open the data file, which is now just filename.
  fd = ::open (filename, O_RDONLY);
  
  if (fd < 0)
    throw Error (FailedSys, "PMDAQFile::open",
		 "failed open(%s)", filename);
  
  if (verbose)
    cerr << "Returning from PMDAQFile::open_file with ndat=" 
         << get_info()->get_ndat() << endl;
}

//! Pads gaps in data
int64_t dsp::PMDAQFile::pad_bytes(unsigned char* buffer, int64_t bytes){
  if( get_info()->get_nbit() != 1 )
    return File::pad_bytes(buffer,bytes);

  // Perhaps this will work- ones and zeros in alternate channels???
  register const unsigned char pad_val = 1 + 4 + 16 + 64;
  for( int64_t i=0; i<bytes; ++i)
    buffer[i] = pad_val;
  
  return bytes;
}

