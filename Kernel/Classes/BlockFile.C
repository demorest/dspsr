/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <sys/stat.h>
#include <sys/types.h>

#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsp/BlockFile.h"

using namespace std;

//! Constructor
dsp::BlockFile::BlockFile (const char* name) : File (name)
{
  block_bytes = 0;
  block_header_bytes = 0;
  block_tailer_bytes = 0;
}

//! Destructor
dsp::BlockFile::~BlockFile ()
{
}

uint64 dsp::BlockFile::get_block_data_bytes() const
{
  if (!block_bytes)
    throw Error (InvalidState, "dsp::BlockFile::get_block_data_bytes",
		 "undefined block size");

  uint64 non_data_bytes = block_header_bytes + block_tailer_bytes;

  if (non_data_bytes > block_bytes)
    throw Error (InvalidState, "dsp::BlockFile::get_block_data_bytes",
		 "block_bytes="UI64" < header+tailer_bytes="UI64,
		 block_bytes, non_data_bytes);

  return block_bytes - non_data_bytes;
}

//! Load bytes from file
int64 dsp::BlockFile::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "dsp::BlockFile::load_bytes() nbytes=" << bytes << endl;

  uint64 block_data_bytes = get_block_data_bytes ();
  uint64 to_load = bytes;

  while (to_load) {
  
    uint64 to_read = block_data_bytes - current_block_byte;
    if (to_read > to_load)
      to_read = to_load;

    ssize_t bytes_read = read (fd, buffer, to_read);
 
    if (bytes_read < 0)
      throw Error (FailedSys, "dsp::BlockFile::load_bytes", "read(%d)", fd);

    to_load -= bytes_read;
    buffer += bytes_read;
    current_block_byte += bytes_read;

    if (current_block_byte == block_data_bytes) {

      skip_extra ();
      current_block_byte = 0;

    }

    // probably the end of file
    if (uint64(bytes_read) < to_read)
      break;
  }

  return bytes - to_load;
}

void dsp::BlockFile::skip_extra ()
{
  if (lseek (fd, block_header_bytes + block_tailer_bytes, SEEK_CUR) < 0)
    throw Error (FailedSys, "dsp::BlockFile::load_bytes", "seek(%d)", fd);
}

//! Adjust the file pointer
int64 dsp::BlockFile::seek_bytes (uint64 nbytes)
{
  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes nbytes=" << nbytes << endl;
  
  if (fd < 0)
    throw Error (InvalidState, "dsp::BlockFile::seek_bytes", "invalid fd");

  uint64 block_data_bytes = get_block_data_bytes ();

  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes block_bytes=" << block_bytes
         << " block_header_bytes=" << block_header_bytes
         << " block_tailer_bytes=" << block_tailer_bytes << endl;

  uint64 current_block = nbytes / block_data_bytes;
  current_block_byte = nbytes % block_data_bytes;

  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes current_block="<< current_block <<endl;

  uint64 tot_header_bytes = (current_block+1) * block_header_bytes;
  uint64 tot_tailer_bytes = current_block * block_tailer_bytes;

  uint64 to_byte = nbytes + header_bytes + tot_header_bytes + tot_tailer_bytes;

  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes SEEK_SET to " << to_byte << endl;

  if (lseek (fd, to_byte, SEEK_SET) < 0)
    throw Error (FailedSys, "dsp::BlockFile::seek_bytes",
		 "lseek ("UI64")", to_byte);

  return nbytes;
}

int64 dsp::BlockFile::fstat_file_ndat (uint64 tailer_bytes)
{
  if (verbose)
    cerr << "dsp::BlockFile::fstat_file_ndat header=" << header_bytes
	 << " block=" << block_bytes 
	 << " data=" << get_block_data_bytes() << endl;

  struct stat file_stats;

  if (fstat(fd, &file_stats) != 0)
    throw Error (FailedCall, "dsp::BlockFile::fstat_file_ndat","fstat(%d)",fd);

  int64 actual_file_sz = file_stats.st_size - header_bytes - tailer_bytes;

  uint64 nblocks = actual_file_sz / block_bytes;
  uint64 extra = actual_file_sz % block_bytes;

  uint64 block_data_bytes = get_block_data_bytes ();

  uint64 data_bytes = nblocks * block_data_bytes;

  if (extra > block_header_bytes) {
    extra -= block_header_bytes;
    if (extra > block_data_bytes)
      extra = block_data_bytes;
    data_bytes += extra;
  }

  return info.get_nsamples (data_bytes);
}
