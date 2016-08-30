/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West & Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/LBADR64_File.h"

#include "Error.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include <iostream>

dsp::LBADR64_File::LBADR64_File (const char* filename) 
  : File ("LBADR64")
{
}

dsp::LBADR64_File::~LBADR64_File ()
{

}

bool dsp::LBADR64_File::is_valid (const char* filename) const
{
  int fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::LBADR64_File::open",
		 "failed fopen(%s)", filename);
 
  char hdr[4096];
  read(fd, hdr, 4096);
 
  ::close(fd);

  // Test for basic header format and 64 MHz recording mode
  //
  if(hdr[13] == ':') {     // Current version, 4096-byte ASCII header
    std::cerr << "lba64: Recognised 4096-byte LBADR header" << std::endl;
    float bw = 0.0;
    std::string strhdr = hdr;
    int pos1 = strhdr.find("BANDWIDTH");
    if (pos1 == std::string::npos) {
      std::cerr << "lba64: Bandwidth keyword missing from header" << std::endl;
      return false;
    }
    sscanf(hdr+pos1, "BANDWIDTH %f", &bw);
    if (bw == 64.0) {
      std::cerr << "lba64: Recorded bandwidth = 64 MHz" << std::endl;
      return true;
    }
    else {
      std::cerr << "lba64: Recorded bandwidth unsupported" << std::endl;
      return false;
    }
  }
  else
    return false;
  
}

void dsp::LBADR64_File::open_file (const char* filename)
{
  
  fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::LBADR64_File::open",
		 "failed fopen(%s)", filename);

  header_bytes = 4096;
  
  read(fd, header, header_bytes);
  
  struct tm date;
  
  char tmp[5];
  tmp[0] = header[5];
  tmp[1] = header[6];
  tmp[2] = header[7];
  tmp[3] = header[8];
  tmp[4] = '\0';
  date.tm_year = atoi(tmp) - 1900;

  tmp[0] = header[9];
  tmp[1] = header[10];
  tmp[2] = '\0';
  tmp[3] = '\0';
  date.tm_mon  = atoi(tmp) - 1;

  tmp[0] = header[11];
  tmp[1] = header[12];
  date.tm_mday = atoi(tmp);

  tmp[0] = header[9];
  tmp[1] = header[10];
  tmp[0] = header[14];
  tmp[1] = header[15];  
  date.tm_hour = atoi(tmp);

  tmp[0] = header[16];
  tmp[1] = header[17];
  date.tm_min  = atoi(tmp);

  tmp[0] = header[18];
  tmp[1] = header[19];  
  date.tm_sec  = atoi(tmp) ;

  utc_t utc;
  
  tm2utc(&utc, date);

  get_info()->set_start_time(utc);

  get_info()->set_nbit(2);
  get_info()->set_npol(2);
  get_info()->set_nchan(1);
  
  get_info()->set_state(Signal::Nyquist);
  get_info()->set_machine("LBADR64");

  // In 64 MHz bandwidth mode, the polarisations are packed in
  // alternate bytes, with 4 consecutive samples per byte

  get_info()->set_rate(128000000);
  get_info()->set_bandwidth(64.0);

  // ////////////////////////////////////////////////////////////////
  // Change this as required. The defaults probably won't be correct!
  // ////////////////////////////////////////////////////////////////

  get_info()->set_centre_frequency(1420.0);

  get_info()->set_telescope( "Hobart" );   // 4 = Hobart, 7 = Parkes, 6 = Tid

  struct stat file_info;
  
  stat (filename, &file_info);
  
  // To begin, file_info.st_size contains number of bytes in file. 
  // Subtract header_bytes, multiply by 8 to get the number of data bits.
  // Divide by the number of bits per sample to get the total number of
  // samples, then divide by the number of channels used to get the total
  // number of unique time samples.

  get_info()->set_ndat( (int64_t((file_info.st_size - header_bytes)) * 8) / 
		 (get_info()->get_nbit()*get_info()->get_npol()*get_info()->get_nchan()) );

  // Set the minimum time unit to be the number of samples per byte
  // (because we work in numbers of whole bytes)

  resolution = 4;
}

//! Pads gaps in data
int64_t dsp::LBADR64_File::pad_bytes(unsigned char* buffer, int64_t bytes){
  if( get_info()->get_nbit() != 2 )
    throw Error(InvalidState,"dsp::LBADR64_File::pad_bytes()",
		"Can only pad if nbit=2.  nbit=%d",get_info()->get_nbit());

  register const unsigned char val = 255;
  for( int64_t i=0; i<bytes; ++i)
    buffer[i] = val;
  
  return bytes;
}
