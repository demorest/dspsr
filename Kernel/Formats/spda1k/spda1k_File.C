/***************************************************************************
 *
 *   Copyright (C) 2009 by Aidan Hotan
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/spda1k_File.h"

#include "Error.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include <iostream>
#include <string>

using namespace std;

dsp::SPDA1K_File::SPDA1K_File (const char* filename) 
  : File ("SPDA1K")
{
}

dsp::SPDA1K_File::~SPDA1K_File ()
{
}

bool dsp::SPDA1K_File::is_valid (const char* filename) const
{
  string useful = filename;
  if(useful.find(".spda1k") != string::npos) {
    return true;
  }
  else {
    return false;
  } 
}

void dsp::SPDA1K_File::open_file (const char* filename)
{
  
  fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::SPDA1K_File::open",
		 "failed fopen(%s)", filename);

  get_info()->set_start_time(55100.02180556);

  get_info()->set_nbit(8);
  get_info()->set_npol(1);
  get_info()->set_nchan(1);
  
  get_info()->set_state(Signal::Nyquist);
  get_info()->set_machine("SPDA1K");

  get_info()->set_rate(250000000);
  get_info()->set_bandwidth(-125.0);

  // ////////////////////////////////////////////////////////////////
  // Change this as required. The defaults probably won't be correct!
  // ////////////////////////////////////////////////////////////////

  get_info()->set_centre_frequency(1062.5);

  get_info()->set_telescope( "Hobart" );

  struct stat file_info;
  
  stat (filename, &file_info);
  
  // To begin, file_info.st_size contains number of bytes in file. 
  // Mltiply by 8 to get the number of data bits.
  // Divide by the number of bits per sample to get the total number of
  // samples, then divide by the number of channels used to get the total
  // number of unique time samples.

  get_info()->set_ndat( (int64_t(file_info.st_size) * 8) / 
		 (get_info()->get_nbit()*get_info()->get_npol()*get_info()->get_nchan()) );

  // Set the minimum time unit to be the number of samples per byte
  // (because we work in numbers of whole bytes)

  resolution = 1;
}

//! Pads gaps in data
int64_t dsp::SPDA1K_File::pad_bytes(unsigned char* buffer, int64_t bytes){
  return bytes;
}
