/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/File.h"
#include "dsp/PseudoFile.h"

#include "Reference.h"
#include "Error.h"
#include "RealTimer.h"
#include "tostring.h"

#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

using namespace std;
using std::cerr;

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

//! Return a pointer to a new instance of the appropriate sub-class
dsp::File* dsp::File::create (const char* filename, int _bs_index)
{ 
  if (verbose)
    std::cerr << "dsp::File::create filename='" << filename 
	      << "' bs_index=" << _bs_index << endl;

  // check if file can be opened for reading
  int fd = ::open (filename, O_RDONLY);
  if (fd < 0) throw Error (FailedSys, "dsp::File::create",
			  "cannot open '%s'", filename);
  ::close (fd);

  if (verbose) std::cerr << "dsp::File::create with " << registry.size() 
			 << " registered sub-classes" << std::endl;

  for (unsigned ichild=0; ichild < registry.size(); ichild++) try
  {
      if (verbose)
	std::cerr << "dsp::File::create testing " 
		  << registry[ichild]->get_name() << endl;;

      if ( registry[ichild]->is_valid (filename,_bs_index) ) {	

	if (verbose)
	  std::cerr << "dsp::File::create " << registry[ichild]->get_name()
		    << "::is_valid() returned true" << endl;

	File* child = registry.create (ichild);
	child->open( filename,_bs_index );	
	return child;	

      }

  }
  catch (Error& error)
  {
    if (verbose)
      std::cerr << "dsp::File::create failed while testing "
                << registry[ichild]->get_name() << endl
                << error.get_message() << endl;
  }
  
  string msg = filename;

  msg += " not a recognized file format\n\t" + tostring(registry.size()) + " registered Formats: ";

  for (unsigned ichild=0; ichild < registry.size(); ichild++)
    msg += registry[ichild]->get_name() + " ";

  throw Error (InvalidParam, "dsp::File::create", msg);
}

void dsp::File::open (const char* filename, int _bs_index)
{
  bs_index = _bs_index;

  if (verbose)
    cerr << "dsp::File::open filename='" << filename << "'" << endl;

  const PseudoFile* null_ptr = 0;
  open (filename, null_ptr );
}

void dsp::File::open (const PseudoFile& file)
{
  if (verbose)
    cerr << "dsp::File::open PseudoFile::filename='" 
	 << file.filename << "'" << endl;

  open (0, &file);
}

void dsp::File::open (const char* filename, const PseudoFile* file)
{
  if (verbose)
    cerr << "dsp::File::open bs_index=" << bs_index << endl;

  close();

  if (filename) {

    open_file (filename);
      
    if (info.get_ndat() == 0)
      set_total_samples ();

  }
  else if (file) {

    close ();

    filename = file->filename.c_str();
    bs_index = file->bs_index;

    fd = ::open (filename, O_RDONLY);
    if (fd < 0)
      throw Error (FailedSys, "dsp::File::open PseudoFile", 
		   "failed open(%s)", filename);
  
    header_bytes = file->header_bytes;
    info = *file;
  }

  current_filename = filename;

  // ensure that file is set to load the first sample after the header
  seek_bytes (0);

  reset ();
}

void dsp::File::close ()
{
  if (fd < 0)
    return;
    
  int err = ::close (fd);
  if (err < 0)
    throw Error (FailedSys, "dsp::File::close", "failed close(%d)", fd);

  fd = -1;
}

void dsp::File::reopen ()
{
  if (fd >= 0)
    throw Error (InvalidState, "dsp::File::reopen", "already open");

  fd = ::open (current_filename.c_str(), O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::File::reopen",
		 "failed open(%s)", current_filename.c_str());

  seek_bytes (0);
}

dsp::PseudoFile dsp::File::get_pseudofile(){
  return PseudoFile(this);
}

void dsp::File::set_total_samples ()
{
  info.set_ndat (fstat_file_ndat());
}

//! Load bytes from file
int64 dsp::File::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "dsp::File::load_bytes nbytes=" << bytes << endl;

  int64 old_pos = lseek(fd,0,SEEK_CUR);

  ssize_t bytes_read = read (fd, buffer, size_t(bytes));
 
  if (bytes_read < 0)
    perror ("dsp::File::load_bytes read error");

  int64 new_pos = lseek(fd,0,SEEK_CUR);
  uint64 end_pos = get_info()->get_nbytes() + uint64(header_bytes);

  if (verbose)
    cerr << "dsp::File::load_bytes bytes_read=" << bytes_read
	 << " old_pos=" << old_pos << " new_pos=" << new_pos 
	 << " end_pos=" << end_pos << endl;

  if( uint64(new_pos) >= end_pos ){
    bytes_read = ssize_t(end_pos - old_pos);
    lseek(fd,end_pos,SEEK_SET);
    end_of_data = true;
  }
  else
    end_of_data = false;

  return bytes_read;
}

//! Adjust the file pointer
int64 dsp::File::seek_bytes (uint64 bytes)
{
  if (verbose)
    cerr << "dsp::File::seek_bytes nbytes=" << bytes 
         << " header_bytes=" << header_bytes << endl;
  
  if (fd < 0)
    throw Error (InvalidState, "dsp::File::seek_bytes", "invalid fd");

  bytes += header_bytes;
  int64 retval = lseek (fd, bytes, SEEK_SET);
  if (retval < 0)
    throw Error (FailedSys, "dsp::File::seek_bytes", "lseek error");

  if( uint64(retval) == get_info()->get_nbytes()+uint64(header_bytes) )
    end_of_data = true;
  else
    end_of_data = false;

  // return absolute data byte offset from the start of file
  return retval - header_bytes;
}

/* Determine the number of time samples from the size of the file */
int64 dsp::File::fstat_file_ndat (uint64 tailer_bytes)
{
  if (fd < 0)
    throw Error (InvalidState, "dsp::File::fstat_file_ndat", "fd < 0");

  struct stat buf;
  if (fstat (fd, &buf) < 0)
    throw Error (FailedSys, "dsp::File::fstat_file_ndat",
                 "fstat(%s)", current_filename.c_str());

  if (uint64(buf.st_size) < header_bytes + tailer_bytes)
    throw Error (InvalidState, "dsp::File::fstat_file_ndat",
                 "file size=%d < header size=%d + tailer_size=%d",
                 buf.st_size, header_bytes, tailer_bytes);

  uint64 total_bytes = buf.st_size - header_bytes - tailer_bytes;

  if( verbose )
    cerr << "dsp::File::fstat_file_ndat(): buf=" << buf.st_size
	 << " header_bytes=" << header_bytes 
	 << " tailer_bytes=" << tailer_bytes
	 << " total_bytes=" << total_bytes << endl;

  return info.get_nsamples (total_bytes);
}

//! Over-ride this function to pad data via HoleyFile
int64 dsp::File::pad_bytes(unsigned char* buffer, int64 bytes){
  throw Error(InvalidState,"dsp::File::pad_bytes()",
	      "This class (%s) doesn't have a pad_bytes() function",get_name().c_str());
  return -1;
}





