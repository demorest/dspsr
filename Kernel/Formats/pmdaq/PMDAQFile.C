#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include "PMDAQFile.h"
#include "PMDAQ_Observation.h"

#include "genutil.h"   // Is this really needed?

// Register the PMDAQFile class with the File::registry
static Registry::List<dsp::File>::Enter<dsp::PMDAQFile> entry;

#define PMDAQ_HEADER_SIZE 648

// Loads header into character array pmdaq_header from file filename.

// Takes root name as the name, ie takes SWT001 and adds .hdr to it.

int dsp::PMDAQFile::get_header (char* pmdaq_header, const char* filename)
{
  char * hdr_name = new char [ strlen(filename)+5];

  if (hdr_name == NULL) {
    fprintf(stderr,"dsp::PMDAQFile::get_header - Error newing hdr_name of %d chars\n", strlen(filename)+5);
    return -1;
  }
  strcpy(hdr_name,filename);

  int fd = ::open (hdr_name, O_RDONLY);

  if (fd < 0) {
    if (verbose)
      fprintf (stderr, "PMDAQFile::get_header - failed open(%s): %s", 
	       hdr_name, strerror(errno));
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
  if (get_header (pmdaq_header, filename) < 0)
    return false;

  // verify that the buffer read contains a valid PMDAQ header
  // First three chars should be "PMD"
  // But is that good enough?

  if (strncmp(&pmdaq_header[4],"PMD",3)!=0) return false;
  if (verbose) printf("File %s is valid PMDAQ data!\n",filename);
  return true;
}

void dsp::PMDAQFile::open (const char* filename)
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

  // Open the data file, which is filename, + .dat

  char * data_name = new char [ strlen(filename)+1];

  if (data_name == NULL) {
    throw_str("dsp::PMDAQFile::open - Error newing data_name of %d chars\n", strlen(filename)+1);
  }

  strcpy(data_name,filename);
  data_name[strlen(filename)-3]='d';
  data_name[strlen(filename)-2]='a';
  data_name[strlen(filename)-1]='t';

  fd = ::open (data_name, O_RDONLY);

  if (fd < 0)
    throw_str ("PMDAQFile::open - failed open(%s): %s", 
	       data_name, strerror(errno));

  // MXB I don't think we need this
    reset();
    absolute_position = 0;

  if (verbose)
    cerr << "Returning from PMDAQFile::open" << endl;
}

// Loads bytes bytes from current location
// Needs to know where you are currently up to.

#define HEADER_BYTES 4
#define DATA_BYTES 48*1024
#define TRAILER_BYTES 4

int64 dsp::PMDAQFile::load_bytes (unsigned char * buffer, uint64 bytes){
  int part_chunk_of_48k;
  int whole_chunks_of_48k;
  int last_chunk_of_48k;

  int prelim_skip;
  int64 loaded_bytes;
  int64 retval;

  loaded_bytes = 0;

  // determine if in invalid data section, if so skip until
  // the end of the header.

  int position = (absolute_position%(HEADER_BYTES+DATA_BYTES+TRAILER_BYTES));

  if (position < HEADER_BYTES) {
    // skip until end of header
    // set absolute_position accordingly
    retval = lseek (fd, HEADER_BYTES, SEEK_CUR);
    if (retval < 0 ) return (loaded_bytes);
    absolute_position += HEADER_BYTES;
  }

  if (position >= HEADER_BYTES+DATA_BYTES) {
    // skip until end of header
    // set absolute_position accordingly
    retval = lseek (fd, TRAILER_BYTES, SEEK_CUR);
    if (retval < 0 ) return (loaded_bytes);
    absolute_position += TRAILER_BYTES;
  }

  part_chunk_of_48k = 0;
  if ((position >= HEADER_BYTES) && (position < HEADER_BYTES+DATA_BYTES)) {
    // Determine how many bytes to load in first chunk.
    if ((int64)bytes> (HEADER_BYTES+DATA_BYTES-position))
      part_chunk_of_48k = HEADER_BYTES + DATA_BYTES - position;
    else
      part_chunk_of_48k = bytes;
  }

  // Load what you can and bomb if you fail.

  retval = read (fd, buffer, part_chunk_of_48k);

  if (retval < 0 ) {
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

  // Determine the number of 48k blocks and load them.

  int nblocks = (bytes-loaded_bytes)/ DATA_BYTES;

  for (int i=0;i<nblocks;i++){

    // Skip mini header
    retval = lseek (fd, HEADER_BYTES, SEEK_CUR);
    if (retval < 0 ) return (loaded_bytes);
    absolute_position += HEADER_BYTES;

    // Load data, and deal with failures
    retval = read (fd, buffer, DATA_BYTES);
    if (retval == -1 ) {
      return (int64) loaded_bytes;
    }

    if (retval < DATA_BYTES) {
      absolute_position += retval;
      loaded_bytes += retval;
      return (int64) loaded_bytes;
    }

    absolute_position += DATA_BYTES;
    loaded_bytes += DATA_BYTES;
    buffer += DATA_BYTES;

    // Skip the trailer
    retval = lseek (fd, TRAILER_BYTES, SEEK_CUR);
    if (retval < 0 ) return (loaded_bytes);
    absolute_position += TRAILER_BYTES;

  }

  if (loaded_bytes == bytes) return (int64) loaded_bytes;

  // Load the last part.

  last_chunk_of_48k = bytes - loaded_bytes;

  // Load and bomb if fails.
  retval = read (fd, buffer, last_chunk_of_48k);

  if (retval < 0 ) return (int64) loaded_bytes;

  absolute_position += retval;
  loaded_bytes += retval;
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

// skip number_to_skip bytes

 int64 retval = lseek (fd, number_to_skip, SEEK_SET);

 if (retval < 0) {
   perror ("dsp::PMDAQFile::seek_bytes lseek error\n");
   absolute_position = 0;
   return -1;
 }

 absolute_position = bytes;
 return (bytes);

}




