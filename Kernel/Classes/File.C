#include <sys/types.h>
#include <unistd.h>

#include "File.h"
#include "Error.h"

Registry::List<dsp::File> dsp::File::registry;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::File* dsp::File::create (const char* filename)
{ 
  try {

    if (verbose) cerr << "File::create with " << registry.size() 
		      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++)
      if ( registry[ichild]->is_valid (filename) ) {

	File* child = registry.create (ichild);
	child-> open( filename );

	return child;

      }
  } catch (Error& error) {
    throw error += "File::create";
  }

  throw Error (FileNotFound, "File::create", "%s not valid", filename);
}


void dsp::File::init()
{
  fd = header_bytes = -1;
}

void dsp::File::close()
{
  if (fd >= 0)
    ::close (fd);
  fd = -1;
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

