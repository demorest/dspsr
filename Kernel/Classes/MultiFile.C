#include <sys/types.h>
#include <unistd.h>

#include "MultiFile.h"

dsp::MultiFile::MultiFile ()
{
  index = 0; 
  offset = 0;
}

//! Load bytes from file
int64 dsp::MultiFile::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "MultiFile::load_bytes nbytes=" << bytes << endl;

  if (index < 0 || index >= (int) fds.size()) {
    cerr << "MultiFile::load_bytes invalid index=" << index << endl;
    return -1;
  }

  uint64 bytes_read = 0;

  while (bytes_read < bytes) {

    if (offset > data_bytes[index]) {
      cerr << "MultiFile::load_bytes invalid offset=" << offset << endl;
      return -1;
    }


    if (offset == data_bytes[index]) {
      // the offset into this file has reached the end of the file
      index ++;

      if (index == (int) fds.size()) {
	end_of_data = true;
	break;
      }

      offset = 0;
    }

    int64 this_read = bytes - bytes_read;

    int64 available = data_bytes[index] - offset;

    if (this_read > available)
      this_read = available;

    int64 did_read = read (fds[index], buffer, this_read);

    if (did_read != this_read) {
      cerr << "MultiFile::load_bytes read " << did_read << " out of "
	   << this_read << " bytes" << endl;
      return -1;
    }

    bytes_read += did_read;
    buffer += did_read;
    offset += did_read;

  }

  return bytes_read;
}

//! Adjust the file pointer
int64 dsp::MultiFile::seek_bytes (uint64 bytes)
{
  if (verbose)
    cerr << "MultiFile::seek_bytes nbytes=" << bytes << endl;

  int64 total_bytes = 0;

  for (index = 0; index < (int) fds.size(); index++) {

    if (bytes < data_bytes[index])
      break;

    bytes -= data_bytes[index];
    total_bytes += data_bytes[index];
  }

  if (index == (int) fds.size()) {
    cerr << "MultiFile::seek_bytes (" << total_bytes + bytes << ")"
      " past end of data" << endl;
    return -1;
  }

  offset = bytes;

  bytes += header_bytes[index];
  int64 seeked = lseek (fds[index], bytes, SEEK_SET);
  if (seeked < 0) {
    perror ("MultiFile::seek_bytes lseek error");
    return -1;
  }

  // note: offset_bytes is now an absolute offset from the start of file
  return total_bytes + seeked - header_bytes[index];
}
