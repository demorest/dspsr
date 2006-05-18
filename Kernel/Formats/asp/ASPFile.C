#include "dsp/ASPFile.h"
#include "asp_params.h"

#include "Error.h"

dsp::ASPFile::ASPFile (const char* filename,const char* headername)
  : BlockFile ("ASP")
{
  stream = 0;
}

dsp::ASPFile::~ASPFile ( )
{

}

int open_read_header (const char* filename, struct asp_params* header)
{
  int fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::ASPFile::open_file", 
		            "open(%s) failed", filename);

  int retval = read (fd, header, sizeof(struct asp_params));
  if (retval < 0) {
    ::close (fd);
    throw Error (FailedSys, "dsp::ASPFile::open_file",
			    "read failed");
  }
  return fd;
}


bool dsp::ASPFile::is_valid (const char* filename, int) const
{
  struct asp_params header;

  fd = open_read_header (filename, &header);
  ::close (fd);

  return true;
}

void dsp::ASPFile::open_file (const char* filename)
{
  struct asp_params header;
    
  fd = open_read_header (filename, &header);

  // ///////////////////////////////////////////////////////////////
  //  NBIT
  //
  int nbit = 0;
  if (ascii_header_get (header,"NBIT","%d",&nbit) < 0)
   throw Error (InvalidParam, "ASPFile::open_file", 
		 "failed read NBIT");
	
  cerr << "NBIT = " << nbit << endl;
  info.set_nbit (8);


  // ///////////////////////////////////////////////////////////////
  //  FANOUT
  //
  int fanout = 0;
  if (ascii_header_get (header,"FANOUT","%d",&fanout) < 0)
   throw Error (InvalidParam, "ASPFile::open_file", 
		 "failed read FANOUT");
	
  cerr << "FANOUT = " << fanout << endl;

  struct VLBA_stream* vlba_stream = 0;

  stream = vlba_stream = VLBA_stream_open (filename, nbit, fanout, 0);

  if (!stream)
    throw Error (InvalidParam, "ASPFile::open_file",
		 "failed VLBA_stream_open");

  fd = vlba_stream->infile;

  // store the file pointer for use during reopen, if necessary
  reopen_seek = lseek (fd, 0, SEEK_CUR);

  // instruct the loader to only take gulps in 32/16 lots of nbits
  // necessary since Mk5 files are written in 64-/32-bit words
  cerr << "TRACKS = " << vlba_stream->tracks << endl;
  Input::resolution = vlba_stream->tracks / nbit;  

  cerr << "NCHAN = " << vlba_stream->nchan/fanout << endl;
  info.set_nchan( vlba_stream->nchan/fanout );	

  cerr << "SAMPRATE = " << vlba_stream->samprate << endl;
  info.set_rate ( vlba_stream->samprate );

  int refmjd = 0;
  if (ascii_header_get (header,"REFMJD","%d",&refmjd) < 0)
   throw Error (InvalidParam, "ASPFile::open_file", 
		 "failed read REFMJD");

  cerr << "REFMJD " << refmjd << endl;
  vlba_stream->mjd += refmjd;

  cerr << "MJD = " << vlba_stream->mjd << endl;
  cerr << "SEC = " << vlba_stream->sec << endl;

  info.set_start_time( MJD(vlba_stream->mjd, vlba_stream->sec, 0) );

  // ///////////////////////////////////////////////////////////////
  // TELESCOPE
  //
	
  char hdrstr[256];
  if (ascii_header_get (header,"TELESCOPE","%s",hdrstr) <0)
    throw Error (InvalidParam, "ASPFile::open_file",
		 "failed read TELESCOPE");

  /* user must specify a telescope whose name is recognised or the telescope
     code */
	
  string tel= hdrstr;
  if ( !strcasecmp (hdrstr, "parkes") || tel == "PKS") 
    info.set_telescope_code (7);
  else if ( !strcasecmp (hdrstr, "GBT") || tel == "GBT")
    info.set_telescope_code (1);
  else if ( !strcasecmp (hdrstr, "westerbork") || tel == "WSRT")
    info.set_telescope_code ('i');
  else {
    cerr << "ASPFile:: Warning using telescope code " << hdrstr[0] << endl;
    info.set_telescope_code (hdrstr[0]);
  }
	
  // ///////////////////////////////////////////////////////////////	
  // SOURCE
  //
  if (ascii_header_get (header, "SOURCE", "%s", hdrstr) < 0)
    throw Error (InvalidParam, "ASPFile::open_file",
		 "failed read SOURCE");

  info.set_source (hdrstr);

  // ///////////////////////////////////////////////////////////////
  // FREQ
  //
  // Note that we assign the CENTRE frequency, not the edge of the band
  double freq;
  if (ascii_header_get (header, "FREQ", "%lf", &freq) < 0)
    throw Error (InvalidParam, "ASPFile::open_file",
		 "failed read FREQ");

  info.set_centre_frequency (freq);
	
  //
  // WvS - flag means that even number of channels are result of FFT
  // info.set_dc_centred(true);

  // ///////////////////////////////////////////////////////////////
  // BW
  //
  double bw;
  if (ascii_header_get (header, "BW", "%lf", &bw) < 0)
    throw Error (InvalidParam, "ASPFile::open_file",
		 "failed read BW");

  info.set_bandwidth (bw);
	
  // ///////////////////////////////////////////////////////////////
  // NPOL
  //	
  //  -- generalise this later
	
  info.set_npol(2);    // read in both polns at once

  // ///////////////////////////////////////////////////////////////	
  // NDIM  --- whether the data are Nyquist or Quadrature sampled
  //
  // VLBA data are Nyquist sampled

  info.set_state (Signal::Nyquist);
	  
  // ///////////////////////////////////////////////////////////////
  // NDAT
  // Compute using BlockFile::fstat_file_ndat
  //
  header_bytes = vlba_stream->startoffset;
  block_bytes = FRAMESIZE;
  block_header_bytes = FRAMESIZE - PAYLOADSIZE;

  set_total_samples();

  header_bytes = block_header_bytes = 0;

  //
  // call this only after setting frequency and telescope
  //
  info.set_default_basis ();
	
  string prefix="tmp";    // what prefix should we assign??
	  
  info.set_identifier(prefix+info.get_default_id() );
  info.set_mode(stringprintf ("%d-bit mode",info.get_nbit() ) );
  info.set_machine("ASP");	
}

/*! Uses Walter's next_frame to take care of the modbits business, then
 copies the result from the VLBA_stream::frame buffer into the buffer
 argument. */
int64 dsp::ASPFile::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose) cerr << "ASPFile::load_bytes nbytes =" << bytes << endl;

  if (!dsp::ASPTwoBitCorrection::can_do( get_info() )) {
    if (verbose) 
      cerr << "ASPFile::load_bytes leave it to VLBA_stream_get_data" << endl;
    return bytes;
  }

  struct VLBA_stream* vlba_stream = (struct VLBA_stream*) stream;

  unsigned char* from = (unsigned char*) vlba_stream->payload;
  unsigned char* to = buffer;

  unsigned offset_word = vlba_stream->read_position;
  unsigned step = vlba_stream->fanout * unsigned(get_info()->get_nbyte()+0.5);

  unsigned bytes_read = 0;

#if _DEBUG
  cerr << "PAYLOADSIZE=" << PAYLOADSIZE << endl;
#endif

  while (bytes_read < bytes) {

    if (offset_word >= PAYLOADSIZE) {

      if (next_frame(vlba_stream) < 0) {
	set_eod (true);
	break;
      }

      from = (unsigned char*) vlba_stream->payload;
      offset_word = 0;

    }

#if _DEBUG
    cerr << "bytes=" << bytes << " bytes_read=" << bytes_read
	 << "\n frame=" << vlba_stream->framenum
	 << " offset=" << offset_word << " step=" << step 
	 << "\n firstvalid=" << vlba_stream->firstvalid
	 << " lastvalid=" << vlba_stream->lastvalid << endl;
#endif

    unsigned max_bytes = bytes - bytes_read;
    unsigned done_bytes = 0;

    unsigned invalid_bytes = 0;

    if (offset_word < (unsigned) vlba_stream->firstvalid) {
      invalid_bytes = (vlba_stream->firstvalid - offset_word) * step;
      offset_word = vlba_stream->firstvalid;
    }
    else if (offset_word > (unsigned) vlba_stream->lastvalid) {
      invalid_bytes = (PAYLOADSIZE - offset_word) * step;
      offset_word = PAYLOADSIZE;
    }

    if (invalid_bytes) {

      if (invalid_bytes > max_bytes)
	invalid_bytes = max_bytes;

#if _DEBUG
      cerr << "  invalid bytes = " << invalid_bytes << endl;
#endif
      memset (to, 0, invalid_bytes);
      done_bytes = invalid_bytes;

    }

    else {

      unsigned last_word = PAYLOADSIZE;
      if (vlba_stream->lastvalid+1 < PAYLOADSIZE)
	last_word = vlba_stream->lastvalid+1;

      unsigned copy_bytes = (last_word - offset_word) * step;

      if (copy_bytes > max_bytes)
	copy_bytes = max_bytes;

#if _DEBUG
      cerr << "  copy bytes = " << copy_bytes << endl;
#endif
      memcpy (to, from, copy_bytes);
      done_bytes = copy_bytes;

    }

    to += done_bytes;
    from += done_bytes;
    bytes_read += done_bytes;
    offset_word += done_bytes / step;

    vlba_stream->read_position = offset_word;

  }

  return bytes_read;
}

int64 dsp::ASPFile::seek_bytes (uint64 nbytes)
{
  if (verbose)
    cerr << "ASPFile::seek_bytes nbytes=" << nbytes << endl;

  if (nbytes != 0)
    throw Error (InvalidState, "ASPFile::seek_bytes", "unsupported");

  return nbytes;
}


void dsp::ASPFile::reopen ()
{
  File::reopen();

  static_cast<struct VLBA_stream*>(stream)->infile = fd;

  if (lseek (fd, reopen_seek, SEEK_SET) < 0)
    throw Error (FailedSys, "dsp::ASPFile::reopen",
		 "failed lseek(%u)", reopen_seek);
}
