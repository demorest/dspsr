#include <sys/stat.h>
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
  close ();
}

void dsp::File::init()
{
  fd = -1;

  header_bytes = 0;

  current_filename = "";

  info.init();
}

void dsp::File::open (const char* filename)
{
  close ();

  open_file (filename);

  if (info.get_ndat() == 0)
    set_total_samples ();

  // ensure that file is set to load the first sample after the header
  seek_bytes (0);

  reset ();

  current_filename = filename;
}


void dsp::File::close()
{
  if (fd >= 0)
    ::close (fd);
  init ();
}

void dsp::File::set_total_samples ()
{
  if (fd < 0)
    throw Error (InvalidState, "dsp::File::set_total_samples", "fd < 0");

  struct stat buf;
  if (fstat (fd, &buf) < 0)
    throw Error (FailedSys, "dsp::File::set_total_samples",
		 "fstat(%s)", current_filename.c_str());

  if (buf.st_size < header_bytes)
    throw Error (InvalidState, "dsp::File::set_total_samples",
		 "file size=%d < header size=%d",
		 buf.st_size, header_bytes);

  uint64 total_bytes = buf.st_size - header_bytes;

  info.set_ndat (info.get_nsamples (total_bytes));
}

//! Load bytes from file
int64 dsp::File::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "dsp::File::load_bytes nbytes=" << bytes << endl;

  ssize_t bytes_read = read (fd, buffer, bytes);
  if (bytes_read < 0)
    perror ("dsp::File::load_bytes read error");

  return bytes_read;
}

//! Adjust the file pointer
int64 dsp::File::seek_bytes (uint64 bytes)
{
  if (verbose)
    cerr << "dsp::File::seek_bytes nbytes=" << bytes << endl;

  if (fd < 0) {
    fprintf (stderr, "dsp::File::seek_bytes invalid fd\n");
    return -1;
  }

  bytes += header_bytes;
  int64 retval = lseek (fd, bytes, SEEK_SET);
  if (retval < 0) {
    perror ("dsp::File::seek_bytes lseek error");
    return -1;
  }

  // return absolute data byte offset from the start of file
  return retval - header_bytes;
}












