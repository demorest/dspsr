#include <sys/types.h>
#include <unistd.h>

#include "dsp/File.h"
#include "Error.h"

//! Constructor
dsp::File::File (const char* name) : Seekable (name)
{ 
  init(); 
}
    
/*! The destructor is defined in the .C file so that the
    Reference::To<BitStream> destructor need not know about the BitStream
    class in the .h file, allowing changes to be made to BitStream without
    forcing the re-compilation of code that uses Input but does not use
    BitStream.
*/
dsp::File::~File ()
{
}

void dsp::File::init()
{
  fd = header_bytes = -1;
  current_filename = "No file open";
}

void dsp::File::close()
{
  if (fd >= 0)
    ::close (fd);
  fd = -1;
  current_filename = "No file open";
}

//! Load bytes from file
int64 dsp::File::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "File::load_bytes nbytes=" << bytes << endl;

  ssize_t bytes_read = read (fd, buffer, bytes);
  if (bytes_read < 0)
    perror ("File::load_bytes read error");

  return bytes_read;
}

//! Adjust the file pointer
int64 dsp::File::seek_bytes (uint64 bytes)
{
  if (verbose)
    cerr << "File::seek_bytes nbytes=" << bytes << endl;

  if (fd < 0) {
    fprintf (stderr, "File::seek_bytes invalid fd\n");
    return -1;
  }

  bytes += header_bytes;
  int64 retval = lseek (fd, bytes, SEEK_SET);
  if (retval < 0) {
    perror ("File::seek_bytes lseek error");
    return -1;
  }

  // return absolute data byte offset from the start of file
  return retval - header_bytes;
}












