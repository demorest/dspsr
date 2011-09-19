/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/OutputFile.h"
#include "dsp/BitSeries.h"

#include "Error.h"

#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

using namespace std;

//! Constructor
dsp::OutputFile::OutputFile (const char* operation_name)
  : Operation (operation_name)
{
  fd = -1;
  header_bytes = 0;
  datestr_pattern = "%Y-%m-%d-%H:%M:%S";
}
    
//! Destructor
dsp::OutputFile::~OutputFile ()
{
  if (fd != -1)
    ::close (fd);
}

void dsp::OutputFile::operation ()
{
  if (fd == -1)
  {
    if (output_filename.empty())
    {
      MJD epoch = input->get_start_time();

      vector<char> buffer (FILENAME_MAX);
      char* filename = &buffer[0];

      if (!epoch.datestr( filename, FILENAME_MAX, datestr_pattern.c_str() ))
	throw Error (FailedCall, "dsp::OutputFile::operation",
		     "error MJD::datestr("+datestr_pattern+")");

      output_filename = filename + get_extension();
    }

    open_file (output_filename);
  }


  unload_bytes (input->get_rawptr(), input->get_nbytes());
}

void dsp::OutputFile::open_file (const char* filename)
{
  int oflag = O_WRONLY | O_CREAT | O_TRUNC | O_EXCL;
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP;

  fd = ::open (filename, oflag, mode);
  if (fd < 0)
    throw Error (FailedSys, "dsp::OutputFile::open_file",
		 "error open("+output_filename+")");

  write_header ();
    
  header_bytes = lseek(fd,0,SEEK_CUR);
}

//! Load nbyte bytes of sampled data from the device into buffer
int64_t dsp::OutputFile::unload_bytes (const void* buffer, uint64_t nbyte)
{
  int64_t written = ::write (fd, buffer, nbyte);

  if (written < (int64_t) nbyte)
  {
    Error error (FailedSys, "dsp::OutputFile::unload_bytes");
    error << "error write(fd=" << fd << ",buf=" << buffer
	  << ",nbyte=" << nbyte;
    throw error;
  }

  return written;
}
