/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADABuffer.h"

#include <iostream>
using namespace std;

//! Constructor
dsp::DADABuffer::DADABuffer ()
  : File ("DADABuffer"),
    ipc(IPCIO_INIT)
{
}
    
//! Construct given a shared memory I/O struct
dsp::DADABuffer::DADABuffer (const ipcio_t& _ipc)
  : File ("DADABuffer")
{
  ipc = _ipc;
}

void dsp::DADABuffer::reset()
{
  end_of_data = false;
  current_sample = 0;
  seek (0,SEEK_END);
  last_load_ndat = 0;
}

//! Returns true if filename appears to name a valid CPSR file
bool dsp::DADABuffer::is_valid (const char* filename, int) const
{
  return false;
}

//! Open the file
void dsp::DADABuffer::open_file (const char* filename)
{
}

//! Load bytes from shared memory
int64 dsp::DADABuffer::load_bytes (unsigned char* buffer, uint64 bytes)
{
  if (verbose)
    cerr << "DADABuffer::load_bytes ipcio_read "
         << bytes << " bytes" << endl;

  int64 bytes_read = ipcio_read (&ipc, (char*)buffer, bytes);
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

  int64 absolute_bytes = ipcio_seek (&ipc, bytes, SEEK_SET);
  if (absolute_bytes < 0)
    cerr << "DADABuffer::seek_bytes error ipcio_seek" << endl;

  return absolute_bytes;
}

void dsp::DADABuffer::seek (int64 offset, int whence)
{
  if (verbose)
    cerr << "dsp::DADABuffer::seek " << offset 
	 << " samples from " << whence << endl;

  if (whence == SEEK_END && offset == 0) {

    if (verbose)
      cerr << "dsp::DADABuffer::seek write_buf=" 
	   << ipcbuf_get_write_count( &(ipc.buf) ) << endl;

    ipc.buf.viewbuf ++;

    if (ipcbuf_get_write_count( &(ipc.buf) ) > ipc.buf.viewbuf)
      ipc.buf.viewbuf = ipcbuf_get_write_count( &(ipc.buf) ) + 1;

    ipc.bytes = 0;
    ipc.curbuf = 0;

    if (verbose)
      cerr << "dsp::DADABuffer::seek viewbuf=" << ipc.buf.viewbuf << endl;

  }
  else {
    Input::seek (offset, whence);
  }
}





