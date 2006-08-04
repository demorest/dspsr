/***************************************************************************
 *
 *   Copyright (C) 2002 by Matthew Bailes
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

#include "genutil.h"
#include "Error.h"
#include "environ.h"

#include "dsp/BitSeries.h"
#include "dsp/Observation.h"
#include "dsp/PMDAQ_Extension.h"
#include "dsp/PMDAQFile.h"
#include "dsp/PMDAQ_Observation.h"

#define PMDAQ_HEADER_SIZE 648

bool dsp::PMDAQFile::using_second_band = false;

//! Destructor
dsp::PMDAQFile::~PMDAQFile(){}

dsp::PMDAQFile::PMDAQFile (const char* filename) 
  : File ("PMDAQ")
{
  if (filename)
    open (filename);

  header_bytes = 4;
  data_bytes = 48*1024;
  trailer_bytes = 4;

  chan_begin = -1;
  chan_end = -1;
}

int dsp::PMDAQFile::get_header (char* pmdaq_header, const char* filename)
{
  ////////////////////////////////////////////////////////
  // See if file has been dumped straight to disk using dd
  {
    int fd = ::open(filename,O_RDONLY);
    if( fd < 0 )
      throw Error(InvalidState,"dsp::PMDAQFile::get_header()",
		  "File '%s' does not even exist!",
		  filename);
    
    if( ::read( fd, pmdaq_header, 640) == 640 ){
      if( pmdaq_header[0] == 'P' && pmdaq_header[1] == 'M' && pmdaq_header[2] == 'D' ){
	header_bytes = 640;
	data_bytes = 48*1024;
	trailer_bytes = 0;
	return 0;
      }
    }
    ::close(fd);
  }

  ////////////////////////////////////////////////////////
  // See if sc_td has been used to get file onto disk
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
  Reference::To<PMDAQFile> dummy = new PMDAQFile;

  if ( dummy->get_header (pmdaq_header, filename) < 0)
    return false;
    
  if (strncmp(&pmdaq_header[4],"PMD",3)!=0 && strncmp(&pmdaq_header[0],"PMD",3)!=0 )
    return false;

  if (verbose) printf("File %s is valid PMDAQ data!\n",filename);
  return true;
}

void dsp::PMDAQFile::work_out_ndat(const char* filename){
  uint64 fsize = filesize(filename);

  uint64 chunk_size = header_bytes+data_bytes+trailer_bytes;

  //  if( fsize%chunk_size )
  //throw Error(InvalidState,"dsp::PMDAQFile::work_out_ndat()",
  //	"File '%s' is not an integer number of blocks.  This is assumed by load_bytes() (fsize="UI64" chunk_size="UI64" remainder="UI64,
  //	filename,fsize,chunk_size,fsize%chunk_size);

  uint64 nblocks = fsize/chunk_size;

  uint64 the_data_bytes = nblocks*data_bytes;

  info.set_ndat( info.get_nsamples(the_data_bytes) );

  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::work_out_ndat(%s) YO have set ndat to "UI64" as data_bytes="UI64" fsize="UI64" chunk_size="UI64" nblocks="UI64"\n",
	    filename,info.get_ndat(),data_bytes,fsize,chunk_size,nblocks);
}

// If user has requested channels that lie in the second observing band on disk, modify the bandwidth and centre frequency of output
void dsp::PMDAQFile::modify_info(PMDAQ_Observation* data){
  fprintf(stderr,"In dsp::PMDAQFile::modify_info() with %d and %d cf=%f bw=%d\n",
	  data->has_two_filters(), using_second_band,
	  info.get_centre_frequency(), info.get_bandwidth());

  if( !data->has_two_filters() )
    return;
  
  if( !using_second_band ){
    set_chan_begin( 0 );
    set_chan_end( data->get_freq1_channels() );
    return;
  }

  set_chan_begin( data->get_freq1_channels() );
  set_chan_end( data->get_freq1_channels() + data->get_freq2_channels() );

  info.set_centre_frequency( data->get_second_centre_frequency() );
  info.set_bandwidth( data->get_second_bandwidth() );

  fprintf(stderr,"In dsp::PMDAQFile::modify_info() have set chan_begin to %d chan_end to %d cf=%f bw=%f\n",
	  chan_begin, chan_end, info.get_centre_frequency(), info.get_bandwidth());
}

void dsp::PMDAQFile::open_file (const char* filename)
{  
  if (get_header (pmdaq_header, filename) < 0)
    throw Error(InvalidState,"PMDAQFile::open",
		"failed get_header(%s): %s",
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
  int64 position = absolute_position % int64(header_bytes+data_bytes+trailer_bytes);
  
  if( position < int64(header_bytes) || position >= int64(header_bytes+data_bytes) )
    return 0;
  
  int64 part_chunk_of_48k = int64(header_bytes+data_bytes) - position;
  
  int64 to_load = min(part_chunk_of_48k,int64(bytes));
  
  int64 ret = ::read (fd, buffer, to_load);

  if( ret != to_load )
    throw Error(FailedCall,"dsp::PMDAQFile::load_partial_chunk()",
		"Failed to read in partial chunk of file (requested "I64" got "I64") (absolute_position="I64" position="I64" part_chunk="I64" to_load="I64")",
		to_load, ret,
		absolute_position, position,
		part_chunk_of_48k, to_load);
  
  absolute_position += to_load;
  buffer += to_load;
  
  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::load_partial_chunk() successfully loaded "I64" bytes bringing absolute_position to "I64"\n",
	    to_load, absolute_position);

  return uint64(to_load);
}

int64 dsp::PMDAQFile::load_bytes (unsigned char * buffer, uint64 bytes)
{
  if( verbose )
    fprintf(stderr,"load_bytes(): got info.cf=%f output.cf=%f Got bytes=min("UI64" , "UI64") = "UI64"\n",
	    info.get_centre_frequency(),
	    output->get_centre_frequency(),
	    bytes,bytes_available(),min(bytes, bytes_available()));

  if( !get_output()->has<PMDAQ_Extension>() && chan_end > chan_begin && chan_begin >= 0 ){
    Reference::To<PMDAQ_Extension> ext = new PMDAQ_Extension;
    get_output()->add( ext );
    ext->set_chan_begin( chan_begin );
    ext->set_chan_end( chan_end );
  }

  bytes = min(bytes, bytes_available());

  if( bytes==0 )
    return cleanup(0);

  // Load first partial chunk
  uint64 bytes_loaded = load_partial_chunk(buffer,bytes);

  if( bytes_loaded==bytes )
    return cleanup(bytes_loaded);

  unsigned nchunks = (bytes-bytes_loaded)/data_bytes;

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
  uint64 file_blocks = info.get_nbytes()/data_bytes;
  uint64 file_size = file_blocks * (data_bytes + header_bytes + trailer_bytes);

  if( absolute_position >= int64(file_size-header_bytes-trailer_bytes) )
    end_of_data = true;
  else
    end_of_data = false;

  return bytes_loaded;
}

uint64 dsp::PMDAQFile::bytes_available(){
  uint64 file_blocks = info.get_nbytes()/data_bytes;
  uint64 file_size = file_blocks * (data_bytes + header_bytes + trailer_bytes);
  
  uint64 partial_chunk_bytes = absolute_position % (data_bytes + header_bytes + trailer_bytes);

  uint64 extra_bytes = 0;

  if( partial_chunk_bytes > header_bytes && partial_chunk_bytes <= header_bytes+data_bytes )
    extra_bytes = data_bytes - (partial_chunk_bytes - header_bytes);

  int64 absolute_bytes_left = file_size - absolute_position;
  int64 blocks_left = absolute_bytes_left/(data_bytes + header_bytes + trailer_bytes);

  int64 bytes_in_blocks_left = blocks_left * data_bytes;

  int64 data_bytes_left = bytes_in_blocks_left + extra_bytes;

  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::bytes_available() Got absolute_position="I64" file_blocks="I64" file_size="I64" partial_chunk_bytes="I64" extra_bytes="I64" absolute_bytes_left="I64"-"I64"="I64" blocks_left="I64" bytes_in_blocks_left="I64" data_bytes_left="I64"\n",
	    absolute_position,
	    file_blocks, file_size, partial_chunk_bytes,
	    extra_bytes, 
	    file_size, absolute_position, absolute_bytes_left,
	    blocks_left, bytes_in_blocks_left,
	    data_bytes_left);

  return data_bytes_left;
}

void dsp::PMDAQFile::seek_ahead(){
  int64 position = absolute_position % int64(header_bytes+data_bytes+trailer_bytes);
  
  int64 to_seek = 0;

  if( position >= int64(header_bytes+data_bytes) )
    to_seek = ( int64(header_bytes+data_bytes+trailer_bytes) - position) + header_bytes;
  else if( position > int64(header_bytes) )
    throw Error(InvalidState,"dsp::PMDAQFile::seek_ahead()",
		"position is in middle of a chunk- it's supposed to be in the header/trailer before calling this routine");
  else if( position == int64(header_bytes) )
    return;
  else
    to_seek = int64(header_bytes) - position;

  if( lseek (fd, to_seek, SEEK_CUR) != absolute_position+to_seek )
    throw Error(FailedCall,"dsp::PMDAQFile::seek_ahead()",
		"failed to seek the correct number of bytes");

  absolute_position += to_seek;
  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::seek_ahead() seeked to get absolute_position="I64"\n",
	    absolute_position);
}

uint64 dsp::PMDAQFile::load_last_chunk(unsigned char*& buffer, uint64 bytes){
  if( read (fd, buffer, bytes) != int64(bytes) )
    throw Error(FailedCall,"dsp::PMDAQFile::load_last_chunk()",
		"Failed to read in last partial chunk of file");
  
  absolute_position += bytes;
  buffer += bytes;

  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::load_last_chunk() now got absolute_position="I64"\n",
	    absolute_position);
  
  return bytes;
}

uint64 dsp::PMDAQFile::load_chunk(unsigned char*& buffer){
  uint64 bytes_read = ::read (fd, buffer, data_bytes);

  if( bytes_read != data_bytes )
    throw Error(FailedCall,"dsp::PMDAQFile::load_chunk()",
		"Failed to read full chunk (only loaded "UI64" bytes of file '%s' filesize="I64" curr_pos="I64")",
		bytes_read,get_filename().c_str(),
		int64(filesize(get_filename().c_str())),
		int64(lseek(fd,0,SEEK_CUR)));
  
  absolute_position += data_bytes;
  buffer += data_bytes;
  
  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::load_chunk() successfully loaded chunk of size "I64" bringing absolute_position to "I64"\n",
	    int64(data_bytes), absolute_position);

  return data_bytes;
}

// Seeks to the "bytes" byte in file allowing for 4byte headers + footers
int64 dsp::PMDAQFile::seek_bytes (uint64 bytes){
  if( verbose )
    fprintf(stderr,"Entered dsp::PMDAQFile::seek_bytes ("UI64")\n",
	    bytes);

  // Work out absolute_position based upon header, data & footer size.
  // Then goto it.
  
  int64 number_of_48ks = bytes / (data_bytes);
  int64 residual = bytes - number_of_48ks * data_bytes;
  
  int64 number_to_skip = number_of_48ks * (data_bytes + header_bytes + trailer_bytes) 
    + header_bytes + residual;
  
  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::seek_bytes ("UI64") number_of_48ks="I64" residual="I64" number_to_skip = "I64" * "I64" + "I64" + "I64"\n",
	    bytes, number_of_48ks, residual, number_of_48ks, data_bytes + header_bytes + trailer_bytes, header_bytes, residual);

  // skip number_to_skip bytes
  
  int64 retval = lseek (fd, number_to_skip, SEEK_SET);
  
  if (retval < 0) {
    perror ("dsp::PMDAQFile::seek_bytes lseek error\n");
    absolute_position = 0;
    return -1;
  }
  
  absolute_position = number_to_skip;
  if( verbose )
    fprintf(stderr,"dsp::PMDAQFile::seek_bytes("UI64") now got absolute_position="I64"\n",
	    bytes, absolute_position);

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



