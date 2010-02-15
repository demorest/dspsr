/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADABuffer.h"
#include "dsp/ASCIIObservation.h"
#include "ascii_header.h"
#include "FilePtr.h"

#include <fstream>
using namespace std;

//! Constructor
dsp::DADABuffer::DADABuffer ()
  : File ("DADABuffer")
{
  hdu = 0;
  passive = false;
}

dsp::DADABuffer::~DADABuffer ()
{
  close ();
}

void dsp::DADABuffer::close ()
{
  if (!hdu)
    return;

  if (!eod())
  {
    /*
       If the stream has not reached end of data, then do not mark the
       header as cleared.  The next few lines work with the implementation
       of dada_hdu_unlock_read and should probably be given a less
       obscure interface.  By not clearing the header, the script that
       runs dspsr can use dbnull to clear the ring buffer of the data
       that caused dspsr to quit before end-of-data.
    */
    if (hdu->header)
    {
      free (hdu->header);
      hdu->header = 0;
    }
  }

  if (!passive && dada_hdu_unlock_read (hdu) < 0)
    cerr << "dsp::DADABuffer::close error during dada_hdu_unlock_read" << endl;

  if (passive && dada_hdu_close_view (hdu) < 0)
    cerr << "dsp::DADABuffer::close error during dada_hdu_close_view" << endl;

  if (dada_hdu_disconnect (hdu) < 0)
    cerr << "dsp::DADABuffer::close error during dada_hdu_disconnect" << endl;

  dada_hdu_destroy (hdu);

  hdu = 0;
}

void dsp::DADABuffer::rewind ()
{
  end_of_data = false;
  current_sample = 0;

  if (!passive)
    seek (0,SEEK_SET);

  last_load_ndat = 0;
}

//! Returns true if filename = DADA
bool dsp::DADABuffer::is_valid (const char* filename) const
{
  FilePtr ptr = fopen (filename, "r");
  if (!ptr)
    return false;

  char first[16];
  fgets (first, 16, ptr);
  first[10] = '\0';

  char expect[16] = "DADA INFO:";

  if (strcmp (first, expect) == 0)
    return true;

  if (verbose)
    cerr << "dsp::DADABuffer::is_valid first 10 characters '" << first << "'"
            " != '" << expect << "'" << endl;

  return false;
}

int ipcio_view_eod (ipcio_t* ipcio, unsigned byte_resolution)
{
  ipcbuf_t* buf = &(ipcio->buf);

#ifdef _DEBUG
  fprintf (stderr, "ipcio_view_eod: write_buf=%"PRIu64"\n", 
	   ipcbuf_get_write_count( buf ) );
#endif

  buf->viewbuf ++;

  if (ipcbuf_get_write_count( buf ) > buf->viewbuf)
    buf->viewbuf = ipcbuf_get_write_count( buf ) + 1;

  ipcio->bytes = 0;
  ipcio->curbuf = 0;

  uint64_t current = ipcio_tell (ipcio);
  uint64_t too_far = current % byte_resolution;
  if (too_far)
  {
    int64_t absolute_bytes = ipcio_seek (ipcio,
					 current + byte_resolution - too_far,
					 SEEK_SET);
    if (absolute_bytes < 0)
      return -1;
  }

  return 0;
}


//! Open the file
void dsp::DADABuffer::open_file (const char* filename)
{ 
  if (verbose)
    cerr << "dsp::DADABuffer::open_file" << endl;

  ifstream input (filename);
  if (!input)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot open INFO file: %s", filename);

  std::string line;
  std::getline (input, line);

  if (line != "DADA INFO:")
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "invalid INFO file (no preamble): %s", filename);

  input >> line;

  if (line != "key")
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "invalid INFO file (no key): %s", filename);

  input >> line;

  key_t key = 0;
  int scanned = 0;

  if (line.length())
    scanned = sscanf (line.c_str(), "%x", &key);

  if (scanned != 1)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "invalid INFO file (no key scanned): %s", filename);

  input >> line;

  passive = line == "viewer";

  if (verbose)
    cerr << "dsp::DADABuffer::open_file key=" << key 
	 << " passive=" << passive << endl;

  if (!hdu)
  {
    /* DADA library logger */
    multilog_t* log = multilog_open ("dspsr", 0);
    multilog_add (log, stderr);

    hdu = dada_hdu_create (log);
  }

  dada_hdu_set_key (hdu, key);

  if (dada_hdu_connect (hdu) < 0)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot connect to DADA ring buffers");

  if (!passive && dada_hdu_lock_read (hdu) < 0)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot lock DADA ring buffer read client status");

  if (passive && dada_hdu_open_view (hdu) < 0)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot open DADA ring buffer for viewing");

  if (dada_hdu_open (hdu) < 0)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot open DADA ring buffers");

  if (verbose)
    cerr << "dsp::DADABuffer::open_file HEADER: size=" 
         << hdu->header_size << " content=\n" << hdu->header << endl;

  info = ASCIIObservation (hdu->header);

  if (ascii_header_get (hdu->header, "RESOLUTION", "%u", &byte_resolution) < 0)
    byte_resolution = 1;

  // the resolution is the _byte_ resolution; convert to _sample_ resolution
  resolution = info.get_nsamples (byte_resolution);
  if (resolution == 0)
    resolution = 1;

  if (passive && ipcio_view_eod (hdu->data_block, byte_resolution) < 0)
    throw Error (FailedCall, "dsp::DADABuffer::open_file",
		 "cannot ipcio_view_eod");

  if (verbose)
    cerr << "dsp::DADABuffer::open_file exit" << endl;
}

//! Load bytes from shared memory
int64_t dsp::DADABuffer::load_bytes (unsigned char* buffer, uint64_t bytes)
{
  if (verbose)
    cerr << "DADABuffer::load_bytes ipcio_read "
         << bytes << " bytes" << endl;

  int64_t bytes_read = ipcio_read (hdu->data_block, (char*)buffer, bytes);
  if (bytes_read < 0)
    cerr << "DADABuffer::load_bytes error ipcio_read" << endl;

  if (verbose)
    cerr << "DADABuffer::load_bytes read " << bytes_read << " bytes" << endl;

  return bytes_read;
}

//! Adjust the shared memory pointer
int64_t dsp::DADABuffer::seek_bytes (uint64_t bytes)
{
  if (verbose)
    cerr << "DADABuffer::seek_bytes ipcio_seek "
         << bytes << " bytes" << endl;

  int64_t absolute_bytes = ipcio_seek (hdu->data_block, bytes, SEEK_SET);
  if (absolute_bytes < 0)
    cerr << "DADABuffer::seek_bytes error ipcio_seek" << endl;

  if (verbose)
    cerr << "DADABuffer::seek_bytes absolute_bytes=" << absolute_bytes << endl;

  return absolute_bytes;
}

void dsp::DADABuffer::seek (int64_t offset, int whence)
{
  if (verbose)
    cerr << "dsp::DADABuffer::seek " << offset 
	 << " samples from whence=" << whence << endl;

  if (passive && whence == SEEK_END && offset == 0)
  {
    if (ipcio_view_eod (hdu->data_block, byte_resolution) < 0)
      throw Error (FailedCall, "dsp::DADABuffer::seek",
		   "cannot ipcio_view_eod");
  }
  else
    Input::seek (offset, whence);
}

//! Ensure that block_size is an integer multiple of resolution
void dsp::DADABuffer::set_block_size (uint64_t _size)
{
  if (resolution > 1)
  {
    uint64_t packets = _size / resolution;
    if (_size % resolution)
      packets ++;
    _size = resolution * packets;
  }
  File::set_block_size (_size);
}

//! End-of-data is defined by primary read client (passive viewer)
bool dsp::DADABuffer::eod()
{
  if (passive)
    return ipcbuf_eod ( &(hdu->data_block->buf) );
  else
    return File::eod();
}

void dsp::DADABuffer::set_total_samples ()
{
}


