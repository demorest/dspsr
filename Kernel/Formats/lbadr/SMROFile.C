/***************************************************************************
 *
 *   Copyright (C) 2004 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "SMRO.h"
#include "dsp/SMROFile.h"

#include "Error.h"

#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include <string>
#include <iostream>

using namespace std;

dsp::SMROFile::SMROFile (const char* filename) 
  : File ("SMRO")
{
  legacy = false;
}

dsp::SMROFile::~SMROFile ()
{

}

bool dsp::SMROFile::is_valid (const char* filename) const
{
  int fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::SMROFile::open",
		 "failed fopen(%s)", filename);
 
  char hdr[4096];
  read(fd, hdr, 4096);

  ::close(fd);

  // Two options, considering archived data
  //
  if(hdr[8] == ':') {       // Legacy version, prior to LBA header upgrade
    return true;
    cerr << "lbadr: Recognised legacy LBADR file format, bandwidth 16 MHz" << endl;
  }
  else if(hdr[13] == ':') { // Current version, 4096-byte ASCII header
    cerr << "lbadr: Recognised 4096-byte LBADR header" << endl;
    float bw = 0.0;
    string strhdr = hdr;
    int pos1 = strhdr.find("BANDWIDTH");
    if (pos1 == string::npos) {
      cerr << "lbadr: Bandwidth keyword missing from header" << endl;
      return false;
    }
    sscanf(hdr+pos1, "BANDWIDTH %f", &bw);
    if (bw == 16.0) {
      cerr << "lbadr: Recorded bandwidth = 16 MHz" << endl;
      return true;
    }
    else {
      cerr << "lbadr: Recorded bandwidth unsupported" << endl;
      return false;
    }
  }
  else
    return false;
}

void dsp::SMROFile::open_file (const char* filename)
{
  
  fd = ::open(filename, O_RDONLY);
  
  if(fd == -1)
    throw Error (FailedSys, "dsp::SMROFile::open",
		 "failed fopen(%s)", filename);

  // First, test for the header format (can't do this in the is_valid
  // routine, because it is const and we couldn't store the answer).

  char teststr[21];
  read(fd, teststr, 20);

  if(teststr[13] == ':') {     // Current version, 4096-byte ASCII header
    legacy = false;
  }
  else if(teststr[8] == ':') { // Legacy version, prior to LBA header upgrade
    legacy = true;
  }
  else
    throw Error (InvalidParam, "dsp::SMROFile::open",
		 "file %s has unknown header format", filename);

  // Rewind the data stream and begin again, now that we know the format

  ::close(fd);
  fd = ::open(filename, O_RDONLY);

  if (legacy) {
    header_bytes = 16;
  }
  else {
    header_bytes = 4096;
  }
  
  read(fd, header, header_bytes);
  
  struct tm date;
  
  char tmp[5];
  if (legacy) {
    tmp[0] = header[0];
    tmp[1] = header[1];
    tmp[2] = header[2];
    tmp[3] = header[3];
  }
  else {
    tmp[0] = header[5];
    tmp[1] = header[6];
    tmp[2] = header[7];
    tmp[3] = header[8];
  }
  tmp[4] = '\0';
  date.tm_year = atoi(tmp) - 1900;

  if (legacy) {
    tmp[0] = header[4];
    tmp[1] = header[5];
  }
  else {
    tmp[0] = header[9];
    tmp[1] = header[10];
  }
  tmp[2] = '\0';
  tmp[3] = '\0';
  date.tm_mon  = atoi(tmp) - 1;

  if (legacy) {
    tmp[0] = header[6];
    tmp[1] = header[7];
  }
  else {
    tmp[0] = header[11];
    tmp[1] = header[12];
  }
  date.tm_mday = atoi(tmp);

  if (legacy) {
    tmp[0] = header[9];
    tmp[1] = header[10];
  }
  else {
    tmp[0] = header[14];
    tmp[1] = header[15];  
  }
  date.tm_hour = atoi(tmp);

  if (legacy) {
    tmp[0] = header[11];
    tmp[1] = header[12];
  }
  else {
    tmp[0] = header[16];
    tmp[1] = header[17];
  }
  date.tm_min  = atoi(tmp);

  if (legacy) {
    tmp[0] = header[13];
    tmp[1] = header[14];
  }
  else {
    tmp[0] = header[18];
    tmp[1] = header[19];  
  }
  date.tm_sec  = atoi(tmp) ;

  utc_t utc;
  
  tm2utc(&utc, date);

  get_info()->set_start_time(utc);
  get_info()->set_nbit(2);

#ifdef CHAN8
  get_info()->set_npol(8);
#endif
#ifdef CHAN4
  get_info()->set_npol(4);
#endif
#ifdef CHAN2
  get_info()->set_npol(2);
#endif

  get_info()->set_nchan(1);
  
  get_info()->set_state(Signal::Nyquist);
  get_info()->set_machine("SMRO");

  // Natively, the LBA DAS outputs 4 channels, which represent orthogonal
  // polarisations from two different frequency bands. In 16 MHz mode, only
  // two of the channels (one frequency band and two polarisations) carries
  // information. The other two are discarded at the recording stage. At the
  // moment, 4 MHz and 32 MHz mode do not work (they require more complicated
  // bit masking and shifting to extract the samples in the correct order).

#ifdef MHZ4
  get_info()->set_rate(8000000);
  get_info()->set_bandwidth(4.0);
#endif

#ifdef MHZ16
  get_info()->set_rate(32000000);
  get_info()->set_bandwidth(-16.0);
#endif

#ifdef MHZ32
  get_info()->set_rate(64000000);
  get_info()->set_bandwidth(32.0);
#endif

  get_info()->set_centre_frequency(1420.0);

  // ///////////////////////////////////////////////////////////////
  // Change this as required. The default probably won't be correct!
  // ///////////////////////////////////////////////////////////////

  get_info()->set_telescope( "Hobart" );   // 4 = Hobart, 7 = Parkes, 6 = Tid

  struct stat file_info;
  
  stat (filename, &file_info);
  
  // To begin, file_info.st_size contains number of bytes in file. 
  //   Subtract header_bytes, multiply by 8 to get the number of data bits.
  //   Divide by the number of bits per sample to get the total number of
  //   samples, then divide by the number of channels used to get the total
  //   number of unique time samples.

  // This needs to be checked and fixed?
  
  get_info()->set_ndat( int64_t((file_info.st_size - header_bytes))* 8 / 
		 (get_info()->get_nbit()*get_info()->get_npol()*get_info()->get_nchan()) );
  
#ifdef CHAN8
  unsigned bits_per_byte = 16;
#endif
#ifdef CHAN4
  unsigned bits_per_byte = 8;
#endif
#ifdef CHAN2
  unsigned bits_per_byte = 4;
#endif

  resolution = bits_per_byte / get_info()->get_nbit();
  if (resolution == 0)
      resolution = 1;

}

//! Pads gaps in data
int64_t dsp::SMROFile::pad_bytes(unsigned char* buffer, int64_t bytes){
  if( get_info()->get_nbit() != 2 )
    throw Error(InvalidState,"dsp::SMROFile::pad_bytes()",
		"Can only pad if nbit=2.  nbit=%d",get_info()->get_nbit());

  register const unsigned char val = 255;
  for( int64_t i=0; i<bytes; ++i)
    buffer[i] = val;
  
  return bytes;
}
