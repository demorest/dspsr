/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADABuffer.h"
#include "dsp/ASCIIObservation.h"
#include "ascii_header.h"

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

  if (!passive && dada_hdu_unlock_read (hdu) < 0)
    cerr << "dsp::DADABuffer::close error during dada_hdu_unlock_read" << endl;

  if (passive && dada_hdu_close_view (hdu) < 0)
    cerr << "dsp::DADABuffer::close error during dada_hdu_close_view" << endl;

  if (dada_hdu_disconnect (hdu) < 0)
    cerr << "dsp::DADABuffer::close error during dada_hdu_disconnect" << endl;

  dada_hdu_destroy (hdu);

  hdu = 0;
}

void dsp::DADABuffer::reset()
{
  end_of_data = false;
  current_sample = 0;
  seek (0,SEEK_SET);
  last_load_ndat = 0;
}

//! Returns true if filename = DADA
bool dsp::DADABuffer::is_valid (const char* filename) const
{
  ifstream input (filename);
  if (!input)
    return false;

  std::string line;
  std::getline (input, line);

  if (line == "DADA INFO:")
    return true;

  return false;
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

  if (verbose)
    cerr << "dsp::DADABuffer::open_file exit" << endl;
}

//! Load bytes from shared memory
int64 dsp::DADABuffer::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "DADABuffer::load_bytes ipcio_read "
         << bytes << " bytes" << endl;

  int64 bytes_read = ipcio_read (hdu->data_block, (char*)buffer, bytes);
  if (bytes_read < 0)
    cerr << "DADABuffer::load_bytes error ipcio_read" << endl;

  if (verbose)
    cerr << "DADABuffer::load_bytes read " << bytes_read << " bytes" << endl;

  return bytes_read;
}

//! Adjust the shared memory pointer
int64 dsp::DADABuffer::seek_bytes (uint64 bytes)
{
  if (verbose)
    cerr << "DADABuffer::seek_bytes ipcio_seek "
         << bytes << " bytes" << endl;

  int64 absolute_bytes = ipcio_seek (hdu->data_block, bytes, SEEK_SET);
  if (absolute_bytes < 0)
    cerr << "DADABuffer::seek_bytes error ipcio_seek" << endl;

  if (verbose)
    cerr << "DADABuffer::seek_bytes absolute_bytes=" << absolute_bytes << endl;

  return absolute_bytes;
}

void dsp::DADABuffer::seek (int64 offset, int whence)
{
  if (verbose)
    cerr << "dsp::DADABuffer::seek " << offset 
	 << " samples from whence=" << whence << endl;

  ipcbuf_t* buf = &(hdu->data_block->buf);

  if (passive && whence == SEEK_END && offset == 0)
  {
    if (verbose)
      cerr << "dsp::DADABuffer::seek passive to end write_buf=" 
	   << ipcbuf_get_write_count( buf ) << endl;

    buf->viewbuf ++;

    if (ipcbuf_get_write_count( buf ) > buf->viewbuf)
      buf->viewbuf = ipcbuf_get_write_count( buf ) + 1;

    hdu->data_block->bytes = 0;
    hdu->data_block->curbuf = 0;

    uint64 current = ipcio_tell (hdu->data_block);

    uint64 too_far = current % byte_resolution;
    if (too_far)
    {
      int64 absolute_bytes = ipcio_seek (hdu->data_block,
					 current + byte_resolution - too_far,
					 SEEK_SET);
      if (absolute_bytes < 0)
	cerr << "DADABuffer::seek passed SEEK_END error ipcio_seek" << endl;
    }

    if (verbose)
      cerr << "dsp::DADABuffer::seek viewbuf=" << buf->viewbuf << endl;

  }
  else
    Input::seek (offset, whence);
}

//! Ensure that block_size is an integer multiple of resolution
void dsp::DADABuffer::set_block_size (uint64 _size)
{
  if (resolution > 1)
  {
    uint64 packets = _size / resolution;
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


