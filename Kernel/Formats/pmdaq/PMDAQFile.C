#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

#include "genutil.h"
#include "Error.h"
#include "environ.h"

#include "dsp/PMDAQFile.h"
#include "dsp/PMDAQ_Observation.h"

#define PMDAQ_HEADER_SIZE 648

#define HEADER_BYTES 4
#define DATA_BYTES (48*1024)
#define TRAILER_BYTES 4

dsp::PMDAQFile::PMDAQFile (const char* filename) 
  : File ("PMDAQ")
{
  using_second_band = false;

  if (filename)
    open (filename);
}

// Loads header into character array pmdaq_header from file filename.

// Takes root name as the name, ie takes SWT001 and adds .hdr to it.

int dsp::PMDAQFile::get_header (char* pmdaq_header, const char* filename)
{
  string str_filename = string (filename);
  int pos = str_filename.find_last_of(".",str_filename.size()-1);
  string hdr_name = str_filename.substr(0,pos) + ".hdr";

  if( verbose )
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

bool dsp::PMDAQFile::is_valid (const char* filename,int) const
{
  if (get_header (pmdaq_header, filename) < 0){
    if(verbose) fprintf(stderr,"false 1\n");
    return false;
  }

  // verify that the buffer read contains a valid PMDAQ header
  // First three chars should be "PMD"
  // But is that good enough?

  if (verbose) 
    fprintf(stderr,"3 chars are '%c' '%c' '%c'\n",
	  pmdaq_header[4],pmdaq_header[5],pmdaq_header[6]);
  
  if (strncmp(&pmdaq_header[4],"PMD",3)!=0){
    fprintf(stderr,"false 2\n");
    return false;
  }

  if (verbose) printf("File %s is valid PMDAQ data!\n",filename);
  return true;
}

void dsp::PMDAQFile::work_out_ndat(const char* filename){
  uint64 fsize = filesize(filename);

  uint64 chunk_size = uint64(HEADER_BYTES+DATA_BYTES+TRAILER_BYTES);

  if( fsize%chunk_size )
    throw Error(InvalidState,"dsp::PMDAQFile::work_out_ndat()",
		"File '%s' is not an integer number of blocks.  This is assumed by load_bytes() (fsize="UI64" chunk_size="UI64" remainder="UI64,
		filename,fsize,chunk_size,fsize%chunk_size);

  uint64 nblocks = fsize/chunk_size;

  uint64 data_bytes = nblocks*DATA_BYTES;

  info.set_ndat( info.get_nsamples(data_bytes) );

  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::work_out_ndat() have set ndat to "UI64"\n",
	    info.get_ndat());
}

// If user has requested channels that lie in the second observing band on disk, modify the bandwidth and centre frequency of output
void dsp::PMDAQFile::modify_info(PMDAQ_Observation* data){
  if( !data->has_two_filters() || !using_second_band )
    return;
  
  info.set_centre_frequency( data->get_second_centre_frequency() );
  info.set_bandwidth( data->get_second_bandwidth() );
}

void dsp::PMDAQFile::open_file (const char* filename)
{  
  if (get_header (pmdaq_header, filename) < 0)
    throw_str ("PMDAQFile::open - failed get_header(%s): %s",
	       filename, strerror(errno));
  
  PMDAQ_Observation data(pmdaq_header);
  
  info = data;
   
  modify_info(&data);

  work_out_ndat(filename);

  // Open the data file, which is now just filename.
  fd = ::open (filename, O_RDONLY);
  
  if (fd < 0)
    throw_str ("PMDAQFile::open - failed open(%s): %s", 
	       filename, strerror(errno));
  
  absolute_position = 0;
  
  if (verbose)
    cerr << "Returning from PMDAQFile::open_file with ndat=" << info.get_ndat() << endl;
}

// Loads bytes bytes from current location
// Needs to know where you are currently up to.

uint64 dsp::PMDAQFile::load_partial_chunk(unsigned char*& buffer, uint64 bytes){
  int position = (absolute_position%(HEADER_BYTES+DATA_BYTES+TRAILER_BYTES));
  
  if( position < HEADER_BYTES || position >= HEADER_BYTES+DATA_BYTES )
    return 0;
  
  int part_chunk_of_48k = HEADER_BYTES+DATA_BYTES - position;
  
  int to_load = min(part_chunk_of_48k,int(bytes));
  
  if( read (fd, buffer, to_load) != to_load )
    throw Error(FailedCall,"dsp::PMDAQFile::load_partial_chunk()",
		"Failed to read in partial chunk of file");
  
  absolute_position += to_load;
  buffer += to_load;
  
  return to_load;
}

int64 dsp::PMDAQFile::load_bytes (unsigned char * buffer, uint64 bytes)
{
  if( verbose )
    fprintf(stderr,"Got bytes=min("UI64" , "UI64") = "UI64"\n",
	    bytes,bytes_available(),min(bytes, bytes_available()));

  bytes = min(bytes, bytes_available());

  if( bytes==0 )
    return cleanup(0);

  // Load first partial chunk
  uint64 bytes_loaded = load_partial_chunk(buffer,bytes);

  if( bytes_loaded==bytes )
    return cleanup(bytes_loaded);

  unsigned nchunks = (bytes-bytes_loaded)/DATA_BYTES;

  // Load full chunks
  for( unsigned ichunk=0; ichunk<nchunks; ichunk++){
    seek_ahead();
    bytes_loaded += load_chunk(buffer);
  }

  if( bytes_loaded==bytes )
    return cleanup(bytes_loaded);

  seek_ahead();

  // Load last partial chunk
  bytes_loaded += load_last_chunk(buffer,bytes-bytes_loaded);

  return cleanup(bytes_loaded);
}

int64 dsp::PMDAQFile::cleanup(uint64 bytes_loaded){
  uint64 file_blocks = info.get_nbytes()/DATA_BYTES;
  uint64 file_size = file_blocks * (DATA_BYTES + HEADER_BYTES + TRAILER_BYTES);

  if( int64(absolute_position) >= int64(file_size-TRAILER_BYTES) )
    end_of_data = true;
  else
    end_of_data = false;

  return bytes_loaded;
}

uint64 dsp::PMDAQFile::bytes_available(){
  uint64 file_blocks = info.get_nbytes()/DATA_BYTES;
  uint64 file_size = file_blocks * (DATA_BYTES + HEADER_BYTES + TRAILER_BYTES);
  
  uint64 partial_chunk_bytes = absolute_position%(DATA_BYTES + HEADER_BYTES + TRAILER_BYTES);

  uint64 extra_bytes = 0;

  if( partial_chunk_bytes > HEADER_BYTES && partial_chunk_bytes <= HEADER_BYTES+DATA_BYTES )
    extra_bytes = DATA_BYTES - (partial_chunk_bytes - HEADER_BYTES);

  uint64 absolute_bytes_left = file_size - absolute_position;
  uint64 blocks_left = absolute_bytes_left/(DATA_BYTES + HEADER_BYTES + TRAILER_BYTES);

  uint64 bytes_in_blocks_left = blocks_left * DATA_BYTES;

  uint64 data_bytes_left = bytes_in_blocks_left + extra_bytes;

  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::bytes_available() Got absolute_position="UI64" file_blocks="UI64" file_size="UI64" partial_chunk_bytes="UI64" extra_bytes="UI64" absolute_bytes_left="UI64" blocks_left="UI64" bytes_in_blocks_left="UI64" data_bytes_left="UI64"\n",
	    absolute_position,
	    file_blocks, file_size, partial_chunk_bytes,
	    extra_bytes, absolute_bytes_left,
	    blocks_left, bytes_in_blocks_left,
	    data_bytes_left);

  return data_bytes_left;
}

void dsp::PMDAQFile::seek_ahead(){
  int position = (absolute_position%(HEADER_BYTES+DATA_BYTES+TRAILER_BYTES));
  
  int to_seek = 0;

  if( position >= HEADER_BYTES+DATA_BYTES )
    to_seek = (HEADER_BYTES+DATA_BYTES+TRAILER_BYTES - position) + HEADER_BYTES;
  else if( position > HEADER_BYTES )
    throw Error(InvalidState,"dsp::PMDAQFile::seek_ahead()",
		"position is in middle of a chunk- it's supposed to be in the header/trailer before calling this routine");
  else if( position == HEADER_BYTES )
    return;
  else
    to_seek = HEADER_BYTES - position;

  if( lseek (fd, to_seek, SEEK_CUR) != absolute_position+to_seek )
    throw Error(FailedCall,"dsp::PMDAQFile::seek_ahead()",
		"failed to seek the correct number of bytes");

  absolute_position += to_seek;
}

uint64 dsp::PMDAQFile::load_last_chunk(unsigned char*& buffer, uint64 bytes){
  if( read (fd, buffer, bytes) != int64(bytes) )
    throw Error(FailedCall,"dsp::PMDAQFile::load_last_chunk()",
		"Failed to read in last partial chunk of file");
  
  absolute_position += bytes;
  buffer += bytes;
  
  return bytes;
}

uint64 dsp::PMDAQFile::load_chunk(unsigned char*& buffer){
  uint64 bytes_read = read (fd, buffer, DATA_BYTES);

  if( bytes_read != DATA_BYTES )
    throw Error(FailedCall,"dsp::PMDAQFile::load_chunk()",
		"Failed to read full chunk (only loaded "UI64" bytes of file '%s' filesize="I64" curr_pos="I64")",
		bytes_read,get_filename().c_str(),
		int64(filesize(get_filename().c_str())),
		int64(lseek(fd,0,SEEK_CUR)));
  
  absolute_position += DATA_BYTES;
  buffer += DATA_BYTES;

  return DATA_BYTES;
}

// Seeks to the "bytes" byte in file allowing for 4byte headers + footers

int64 dsp::PMDAQFile::seek_bytes (uint64 bytes){
  // Work out absolute_position based upon header, data & footer size.
  // Then goto it.
  
  int number_of_48ks = bytes / (DATA_BYTES);
  int residual = bytes - number_of_48ks * DATA_BYTES;
  
  int64 number_to_skip = number_of_48ks * (DATA_BYTES + HEADER_BYTES + TRAILER_BYTES) 
    + HEADER_BYTES + residual;
  
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

//! Pads gaps in data
int64 dsp::PMDAQFile::pad_bytes(unsigned char* buffer, int64 bytes){
  if( get_info()->get_nbit() != 1 )
    return File::pad_bytes(buffer,bytes);

  // Perhaps this will work- ones and zeros in alternate channels???
  register const unsigned char pad_val = 1 + 4 + 16 + 64;
  for( int64 i=0; i<bytes; ++i)
    buffer[i] = pad_val;
  
  return bytes;
}



