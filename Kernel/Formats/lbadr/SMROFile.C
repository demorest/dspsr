#include "SMRO.h"
#include "dsp/SMROFile.h"

#include "Error.h"
#include "string_utils.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

dsp::SMROFile::SMROFile (const char* filename) 
  : File ("SMRO")
{

  
}

dsp::SMROFile::~SMROFile ()
{

}

bool dsp::SMROFile::is_valid (const char* filename, int) const
{
  int fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::SMROFile::open",
		 "failed fopen(%s)", filename);
  
  char hdr[17];
  read(fd, hdr, 16);
  
  // VERY HASTY TEST...
  if(hdr[8] == ':')
    return true;
  else
    return false;
  
}

void dsp::SMROFile::open_file (const char* filename)
{
  
  fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::SMROFile::open",
		 "failed fopen(%s)", filename);
  
  read(fd, timestamp, 16);

  header_bytes = 16;
  
  struct tm date;
  
  char tmp[5];
  tmp[0] = timestamp[0];
  tmp[1] = timestamp[1];
  tmp[2] = timestamp[2];
  tmp[3] = timestamp[3];
  tmp[4] = '\0';
  date.tm_year = atoi(tmp) - 1900;

  tmp[0] = timestamp[4];
  tmp[1] = timestamp[5];
  tmp[2] = '\0';
  tmp[3] = '\0';
  date.tm_mon  = atoi(tmp) - 1;

  tmp[0] = timestamp[6];
  tmp[1] = timestamp[7];  
  date.tm_mday = atoi(tmp);

  tmp[0] = timestamp[9];
  tmp[1] = timestamp[10];  
  date.tm_hour = atoi(tmp);

  tmp[0] = timestamp[11];
  tmp[1] = timestamp[12];  
  date.tm_min  = atoi(tmp);

  tmp[0] = timestamp[13];
  tmp[1] = timestamp[14];  
  date.tm_sec  = atoi(tmp) ;

  utc_t utc;
  
  tm2utc(&utc, date);


  info.set_start_time(utc);
  info.set_nbit(2);
#ifdef CHAN8
  info.set_npol(8);
#endif
#ifdef CHAN4
  info.set_npol(4);
#endif
#ifdef CHAN2
  info.set_npol(2);
#endif
  info.set_nchan(1);
  
  info.set_state(Signal::Nyquist);
  info.set_machine("SMRO");

#ifdef MHZ32
  info.set_rate(64000000);
  info.set_bandwidth(32.0);
#endif

#ifdef MHZ4
  info.set_rate(8000000);
  info.set_bandwidth(4.0);
#endif

#ifdef MHZ16
  info.set_rate(32000000);
  info.set_bandwidth(16.0);
#endif

  //info.set_centre_frequency(1384.0);
  info.set_centre_frequency(2282.0);

  info.set_telescope_code('2');   // 7 for parkes, 6 for tid, 2 for CAT
  info.set_identifier("v" + info.get_default_id());

  struct stat file_info;
  
  stat (filename, &file_info);
  
  // file_info.st_size contains number of bytes in file, subtract header_bytes (16bytes)
  // This needs to be checked and fixed
  
  info.set_ndat( int64((file_info.st_size - header_bytes) )* 8 / (info.get_nbit()*info.get_npol()*info.get_nchan()) );
  
#ifdef CHAN8
  unsigned bits_per_byte = 16;
#endif
#ifdef CHAN4
  unsigned bits_per_byte = 8;
#endif
#ifdef CHAN2
  unsigned bits_per_byte = 4;
#endif
  resolution = bits_per_byte / info.get_nbit();
  if (resolution == 0)
      resolution = 1;

}

//! Pads gaps in data
int64 dsp::SMROFile::pad_bytes(unsigned char* buffer, int64 bytes){
  if( get_info()->get_nbit() != 2 )
    throw Error(InvalidState,"dsp::SMROFile::pad_bytes()",
		"Can only pad if nbit=2.  nbit=%d",get_info()->get_nbit());

  register const unsigned char val = 255;
  for( int64 i=0; i<bytes; ++i)
    buffer[i] = val;
  
  return bytes;
}
