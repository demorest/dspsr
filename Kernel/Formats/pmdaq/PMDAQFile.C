#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include "dsp/PMDAQFile.h"
#include "dsp/PMDAQ_Observation.h"

#include "genutil.h"   // Is this really needed?

#define PMDAQ_HEADER_SIZE 648

dsp::PMDAQFile::PMDAQFile (const char* filename) 
  : File ("PMDAQ")
{
  if (filename)
    open (filename);
}

// Loads header into character array pmdaq_header from file filename.

// Takes root name as the name, ie takes SWT001 and adds .hdr to it.

int dsp::PMDAQFile::get_header (char* pmdaq_header, const char* filename)
{

  string str_filename = string (filename);
  int pos = str_filename.find_first_of(".",0);
  string hdr_name = str_filename.substr(0,pos) + ".hdr";

  cerr << "dsp::PMDAQFile::get_header looks for header called " <<
    hdr_name << endl;

  int fd = ::open (hdr_name.c_str(), O_RDONLY);

  if (fd < 0) {
    if (verbose)
      fprintf (stderr, "PMDAQFile::get_header - failed open(%s): %s", 
	       hdr_name.c_str(), strerror(errno));
    return -1;
  }

  int retval = read (fd, pmdaq_header, PMDAQ_HEADER_SIZE);

  ::close (fd);    

  if (retval < PMDAQ_HEADER_SIZE) {
    if (verbose)
      fprintf (stderr, "PMDAQFile::get_header - failed read: %s",
	       strerror(errno));

    return -1;
  }
  return 0;
}

static char pmdaq_header [PMDAQ_HEADER_SIZE];

bool dsp::PMDAQFile::is_valid (const char* filename) const
{
  if (get_header (pmdaq_header, filename) < 0){
    if(verbose) fprintf(stderr,"false 1\n");
    return false;
  }

  // verify that the buffer read contains a valid PMDAQ header
  // First three chars should be "PMD"
  // But is that good enough?

  if (verbose) {
    cout << "The 3 chars are: ";
    cout << char(pmdaq_header[4])<<pmdaq_header[5]<<pmdaq_header[6] << endl;
    fprintf(stderr,"3 chars are '%c' '%c' '%c'\n",
	  pmdaq_header[4],pmdaq_header[5],pmdaq_header[6]);
  }
  if (strncmp(&pmdaq_header[4],"PMD",3)!=0){
    fprintf(stderr,"false 2\n");
    return false;
  }

  if (verbose) printf("File %s is valid PMDAQ data!\n",filename);
  return true;
}

void dsp::PMDAQFile::open_file (const char* filename)
{  
  if (get_header (pmdaq_header, filename) < 0)
    throw_str ("PMDAQFile::open - failed get_header(%s): %s",
	       filename, strerror(errno));
  
  // MXB - what have we here?
  PMDAQ_Observation data (pmdaq_header);

  //  Yamasaki verification not needed for PMDAQ data
  //  if (yamasaki_verify (filename, data.offset_bytes, PMDAQ_HEADER_SIZE) < 0)
  //  throw_str ("cpsr2_Construct: YAMASAKI verification failed");

  info = data;

  // Open the data file, which is now just filename.

  string data_name = string(filename);

  fd = ::open (data_name.c_str(), O_RDONLY);

  if (fd < 0)
    throw_str ("PMDAQFile::open - failed open(%s): %s", 
	       data_name.c_str(), strerror(errno));

  absolute_position = 0;

  if (verbose)
    cerr << "Returning from PMDAQFile::open" << endl;
}

// Loads bytes bytes from current location
// Needs to know where you are currently up to.

#define HEADER_BYTES 4
#define DATA_BYTES (48*1024)
#define TRAILER_BYTES 4

int64 dsp::PMDAQFile::load_bytes (unsigned char * buffer, uint64 bytes)
{
  verbose = true;

  int part_chunk_of_48k;
  int last_chunk_of_48k;

  uint64 loaded_bytes = 0;
  int64 retval;

  // determine if in invalid data section, if so skip until
  // the end of the header if loading a partial block

  int position = (absolute_position%(HEADER_BYTES+DATA_BYTES+TRAILER_BYTES));

  //if (verbose) cerr << "absolute position is "<< absolute_position << endl;
  //if (verbose) cerr << "         position is "<< position << endl;

  if ((position < HEADER_BYTES) && (bytes < DATA_BYTES)) {
    // skip until end of header
    // set absolute_position accordingly
    if (verbose) cerr << " dsp::PMDAQFILE::load_bytes skipping header " << endl; 
    retval = lseek (fd, HEADER_BYTES, SEEK_CUR);
    if (retval < 0 ) return (loaded_bytes);
    absolute_position += HEADER_BYTES;
  }

  if (position >= HEADER_BYTES+DATA_BYTES) {
    // skip until end of header
    // set absolute_position accordingly
    if (verbose) cerr << " dsp::PMDAQFILE::load_bytes skipping trailer " << endl; 
    retval = lseek (fd, TRAILER_BYTES, SEEK_CUR);
    if (retval < 0 ) return (loaded_bytes);
    absolute_position += TRAILER_BYTES;
  }

  part_chunk_of_48k = 0;
  
  if( verbose )
    fprintf(stderr,"position=%d HEADER_BYTES=%d DATA_BYTES=%d\n",
	    position,HEADER_BYTES,DATA_BYTES);

  if ((position > HEADER_BYTES) && (position < HEADER_BYTES+DATA_BYTES)) {
    // Determine how many bytes to load in first chunk.
    if ((int64)bytes> (HEADER_BYTES+DATA_BYTES-position))
      part_chunk_of_48k = HEADER_BYTES + DATA_BYTES - position;
    else
      part_chunk_of_48k = bytes;
  }
  if (verbose) cerr << " dsp::PMDAQFILE::load_bytes part_chunk " << part_chunk_of_48k << endl; 

  // Load what you can and bomb if you fail if part_chunk_of_48k is non-zero

  if (part_chunk_of_48k !=0 ) {
    if( verbose )
      fprintf(stderr,"Going to read %d samps from buffer\n",
	      part_chunk_of_48k);
    retval = read (fd, buffer, part_chunk_of_48k);

    if (retval < 0 ) {
      end_of_data = true;
      return (int64) loaded_bytes;
    }

    if (retval != part_chunk_of_48k) {
      absolute_position += retval;
      loaded_bytes += retval;
      return (int64) loaded_bytes;
    }

    if (retval == (int64) bytes) {
      loaded_bytes = retval;
      absolute_position += loaded_bytes;
      return (int64) loaded_bytes;
    }

    buffer += retval;
  }
  // Determine the number of 48k blocks and load them.

  int nblocks = (bytes-loaded_bytes)/ DATA_BYTES;

  if (verbose) cerr << "loading "<< nblocks << " blocks of PMDAQ data " << endl;

  for (int i=0;i<nblocks;i++){

    //if (verbose) cerr << "dsp::PMDAQFile::load_bytes skipping mini header " << endl;
    // Skip mini header
    retval = lseek (fd, HEADER_BYTES, SEEK_CUR);
    if (retval != HEADER_BYTES+absolute_position ) {
      cerr << "dsp::PMDAQFile::load_bytes Error seeking past header retval=\n"
	   << retval << " absolute_position " << absolute_position << " HEADER_BYTES=" << HEADER_BYTES << endl;
      end_of_data = true;
      return (loaded_bytes);
    }
    //if (verbose) cerr << " dsp::PMDAQFILE::load_bytes skipped header " << endl; 
    absolute_position += HEADER_BYTES;

    // Load data, and deal with failures
    retval = read (fd, buffer, DATA_BYTES);
    if (retval == -1 ) {
      cerr << "dsp::PMDAQFile::load_bytes Error reading " << DATA_BYTES <<
	" from file " << endl;
      perror("dsp::PMDAQFile::load_bytes");
      end_of_data = true;
      return (int64) loaded_bytes;
    }

    if (retval < DATA_BYTES) {
      cerr << "dsp::PMDAQFile::load_bytes Only read " << retval <<
	" bytes from file " << endl;
      absolute_position += retval;
      loaded_bytes += retval;
      end_of_data = true;
      return (int64) loaded_bytes;
    }

    absolute_position += DATA_BYTES;
    loaded_bytes += DATA_BYTES;
    buffer += DATA_BYTES;

    // Skip the trailer
    retval = lseek (fd, TRAILER_BYTES, SEEK_CUR);
    if (retval < 0 ) {
      end_of_data = true;
      return (loaded_bytes);
    }
    //if (verbose) cerr << " dsp::PMDAQFILE::load_bytes skipped trailer " << endl; 
    absolute_position += TRAILER_BYTES;

  }

  if (loaded_bytes == bytes) return (int64) loaded_bytes;

  // Load the last part.

  last_chunk_of_48k = bytes - loaded_bytes;
  if (verbose) cerr << " dsp::PMDAQFILE::load_bytes loading partial chunk " << endl; 

  // Load and bomb if fails.
  retval = read (fd, buffer, last_chunk_of_48k);

  if (retval < 0 ) {
    return (int64) loaded_bytes;
  }

  absolute_position += retval;
  loaded_bytes += retval;
  if (retval != last_chunk_of_48k) end_of_data = true;
  return (int64) loaded_bytes;
}

// Seeks to the "bytes" byte in file allowing for 4byte headers + footers

int64 dsp::PMDAQFile::seek_bytes (uint64 bytes){

// Work out absolute_position based upon header, data & footer size.
// Then goto it.

int number_of_48ks = bytes / (DATA_BYTES);
int residual = bytes - number_of_48ks * 48 * 1024;

int64 number_to_skip = number_of_48ks * (48*1024 + HEADER_BYTES +
TRAILER_BYTES) + HEADER_BYTES + residual;

 cerr <<"dsp::PMDAQFile::seek_bytes  WARNING!! Code is completely untested"<<endl;

// skip number_to_skip bytes

 int64 retval = lseek (fd, number_to_skip, SEEK_SET);

 if (retval < 0) {
   perror ("dsp::PMDAQFile::seek_bytes lseek error\n");
   absolute_position = 0;
   return -1;
 }

 absolute_position = number_to_skip;
 return (bytes);

}




