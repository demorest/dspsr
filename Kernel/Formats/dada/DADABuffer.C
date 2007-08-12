/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADABuffer.h"
#include "dsp/ASCIIObservation.h"

#include <iostream>
using namespace std;

//! Constructor
dsp::DADABuffer::DADABuffer ()
  : File ("DADABuffer")
{
  hdu = 0;
}
    
void dsp::DADABuffer::reset()
{
  end_of_data = false;
  current_sample = 0;
  seek (0,SEEK_END);
  last_load_ndat = 0;
}

//! Returns true if filename = DADA
bool dsp::DADABuffer::is_valid (const char* filename, int) const
{
  return string(filename) == "DADA";
}

//! Open the file
void dsp::DADABuffer::open_file (const char* filename)
{
  if (!hdu)
    hdu = dada_hdu_create (NULL);

  if (dada_hdu_connect (hdu) < 0)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot connect to DADA ring buffers");

  if (dada_hdu_lock_read (hdu) < 0)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot lock DADA ring buffer read client status");

  if (dada_hdu_open (hdu) < 0)
    throw Error (InvalidState, "dsp::DADABuffer::open_file",
		 "cannot open DADA ring buffers");

  info = ASCIIObservation (hdu->header);
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
    cerr << "DADABuffer::load_bytes " << bytes << " bytes" << endl;

  return bytes_read;
}

//! Adjust the shared memory pointer
int64 dsp::DADABuffer::seek_bytes (uint64 bytes)
{
  if (verbose)
    cerr << "DADABuffer::load_bytes ipcio_seek "
         << bytes << " bytes" << endl;

  int64 absolute_bytes = ipcio_seek (hdu->data_block, bytes, SEEK_SET);
  if (absolute_bytes < 0)
    cerr << "DADABuffer::seek_bytes error ipcio_seek" << endl;

  return absolute_bytes;
}

void dsp::DADABuffer::seek (int64 offset, int whence)
{
  if (verbose)
    cerr << "dsp::DADABuffer::seek " << offset 
	 << " samples from " << whence << endl;

  ipcbuf_t* buf = &(hdu->data_block->buf);

  if (whence == SEEK_END && offset == 0) {

    if (verbose)
      cerr << "dsp::DADABuffer::seek write_buf=" 
	   << ipcbuf_get_write_count( buf ) << endl;

    buf->viewbuf ++;

    if (ipcbuf_get_write_count( buf ) > buf->viewbuf)
      buf->viewbuf = ipcbuf_get_write_count( buf ) + 1;

    hdu->data_block->bytes = 0;
    hdu->data_block->curbuf = 0;

    if (verbose)
      cerr << "dsp::DADABuffer::seek viewbuf=" << buf->viewbuf << endl;

  }
  else {
    Input::seek (offset, whence);
  }
}





