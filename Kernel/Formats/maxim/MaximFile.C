#include "dsp/MaximFile.h"

#include "Error.h"
#include "string_utils.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

dsp::MaximFile::MaximFile (const char* filename) 
  : File ("Maxim")
{
}

dsp::MaximFile::~MaximFile ()
{
}

bool dsp::MaximFile::is_valid (const char* filename, int) const
{
  int fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::MaximFile::open",
		 "failed fopen(%s)", filename);
  
  char hdr[17];
  read(fd, hdr, 16);
  
  // VERY HASTY TEST... Assumes same format as SMROFile
  if(hdr[8] == ':')
    return true;
  else
    return false;
  
}

void dsp::MaximFile::open_file (const char* filename)
{
  
  fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::MaximFile::open",
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
  info.set_nbit(8);
  info.set_npol(2);
  info.set_nchan(2);
  
  info.set_state(Signal::Nyquist);

  info.set_rate(40000000);
  info.set_machine("Maxim");
  info.set_bandwidth(40.0);

  info.set_centre_frequency(635.0);

  info.set_telescope_code('4'); // Check this... Make sure it IS the 14m
  info.set_identifier("v" + info.get_default_id());

  struct stat file_info;
  
  stat (filename, &file_info);
  
  // file_info.st_size contains number of bytes in file, 
  // subtract header_bytes (16bytes).

  // This needs to be checked and fixed
  
  info.set_ndat( int64((file_info.st_size - header_bytes)/info.get_npol() )* 
		 16 / (info.get_nbit()*info.get_npol()) );
  
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / info.get_nbit();
  if (resolution == 0)
      resolution = 1;

}
