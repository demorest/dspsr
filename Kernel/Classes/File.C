#include "dsp/File.h"
#include "dsp/PseudoFile.h"

#include "Reference.h"
#include "Error.h"
#include "RealTimer.h"
#include "genutil.h"

#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

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
  if( verbose )
    fprintf(stderr,"In dsp::File::create() with filename='%s' and _bs_index=%d\n",
	    filename,_bs_index);

  // check if file can be opened for reading
  FILE* fptr = fopen (filename, "r");
  if (!fptr) throw Error (FailedSys, "dsp::File::create",
			  "cannot open '%s'", filename);
  fclose (fptr);

  try {

    if (verbose) cerr << "dsp::File::create with " << registry.size() 
		      << " registered sub-classes" << endl;

    for (unsigned ichild=0; ichild < registry.size(); ichild++) {
      if( verbose )
	fprintf(stderr,"Testing child %d: '%s'\n",
		ichild,registry[ichild]->get_name().c_str());

      if ( registry[ichild]->is_valid (filename,_bs_index) ) {	
	if( verbose )
	  fprintf(stderr,"%s::is_valid() returned true\n",
		  registry[ichild]->get_name().c_str());
	File* child = registry.create (ichild);
	child->open( filename,_bs_index );	
	return child;	
      }
    }
    
  } catch (Error& error) {
    throw error += "dsp::File::create";
  }
  
  string msg = filename;

  msg += " not a recognized file format\n\tRegistered Formats: ";

  for (unsigned ichild=0; ichild < registry.size(); ichild++)
    msg += registry[ichild]->get_name() + " ";

  throw Error (InvalidParam, "dsp::File::create", msg);
}

void dsp::File::open (const char* filename,int _bs_index)
{
  bs_index = _bs_index;

  if( verbose )
    fprintf(stderr,"\nIn dsp::File::open (char*) with filename='%s' and bs_index=%d\n",
	    filename,bs_index);

  const PseudoFile* null_ptr = 0;
  open (filename, null_ptr );
}

void dsp::File::open (const PseudoFile& file)
{
  if( verbose )
    fprintf(stderr,"\nIn dsp::File::open (pseudofile) with filename='%s'\n",
	    file.filename.c_str());

  open (0, &file);
}

void dsp::File::open (const char* filename, const PseudoFile* file)
{
  if( verbose )
    fprintf(stderr,"In dsp::File::open() with bs_index=%d\n",bs_index);

  close ();

  if (filename) {

    open_file (filename);
      
    if (info.get_ndat() == 0)
      set_total_samples ();

  }
  else if (file) {

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

void dsp::File::close()
{
  if (fd >= 0)
    ::close (fd);
  init ();
}

dsp::PseudoFile dsp::File::get_pseudofile(){
  return PseudoFile(this);
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

  uint64 old_pos = uint64(lseek(fd,0,SEEK_CUR));

  ssize_t bytes_read = read (fd, buffer, bytes);
 
  if (bytes_read < 0)
    perror ("dsp::File::load_bytes read error");

  if( uint64(lseek(fd,0,SEEK_CUR)) >= get_info()->get_nbytes()+uint64(header_bytes) ){
    bytes_read = get_info()->get_nbytes()+uint64(header_bytes) - old_pos;
    lseek(fd,get_info()->get_nbytes()+uint64(header_bytes),SEEK_SET);
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
    cerr << "dsp::File::seek_bytes nbytes=" << bytes << endl;

  if (fd < 0) {
    fprintf (stderr, "dsp::File::seek_bytes invalid fd\n");
    return -1;
  }

  bytes += header_bytes;
  int64 retval = lseek (fd, bytes, SEEK_SET);
  if (retval < 0) {
    perror ("dsp::File::seek_bytes lseek error");
    fprintf(stderr,"Called lseek(%d,"UI64",SEEK_SET)\n",fd,bytes);
    return -1;
  }

  if( uint64(retval) == get_info()->get_nbytes()+uint64(header_bytes) )
    end_of_data = true;
  else
    end_of_data = false;

  // return absolute data byte offset from the start of file
  return retval - header_bytes;
}

/* Do an fstat on the current file descriptor to see if purported ndat is correct */
int64 dsp::File::fstat_file_ndat(uint64 tailer_bytes){
  struct stat file_stats;

  int ret = fstat(fd, &file_stats);

  if( ret!=0 )
    throw Error(FailedCall,"dsp::File::fstat_file_ndat()",
		"fstat on file failed.  Return value=%d\n",ret);

  int64 actual_file_sz = file_stats.st_size;

  int64 bytes_per_samp = info.get_nchan()*info.get_npol()*info.get_ndim()*info.get_nbit()/8;
  
  return (actual_file_sz - header_bytes - tailer_bytes)/bytes_per_samp;
}






