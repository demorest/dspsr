/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/ASPFile.h"
#include "asp_params.h"
#include "data2rcv.h"

#include "Error.h"

#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "machine_endian.h"

using namespace std;

dsp::ASPFile::ASPFile (const char* filename,const char* headername)
  : BlockFile ("ASP")
{
}

dsp::ASPFile::~ASPFile ( )
{
}

void correct_endian (struct asp_params* header)
{
  FromLittleEndian (header->n_ds);
  FromLittleEndian (header->n_chan);
  FromLittleEndian (header->ch_bw);
  FromLittleEndian (header->rf);
  FromLittleEndian (header->band_dir);
  FromLittleEndian (header->dm);
  FromLittleEndian (header->fft_len);
  FromLittleEndian (header->overlap);
  FromLittleEndian (header->n_bins);
  FromLittleEndian (header->t_dump);
  FromLittleEndian (header->n_dump);
  FromLittleEndian (header->n_samp_dump);
  FromLittleEndian (header->imjd);
  FromLittleEndian (header->fmjd);
  FromLittleEndian (header->cal_scan);

  FromLittleEndian (header->ra);
  FromLittleEndian (header->dec);
  FromLittleEndian (header->epoch);
}

void correct_endian (struct data2rcv* block)
{
  FromLittleEndian (block->totalsize);
  FromLittleEndian (block->NPtsSend);
  FromLittleEndian (block->iMJD);
  FromLittleEndian (block->fMJD);
  FromLittleEndian (block->ipts1);
  FromLittleEndian (block->ipts2);
  FromLittleEndian (block->FreqChanNo);
}

int open_read_header (const char* filename,
                      struct asp_params* header,
                      struct data2rcv* block)
{
  int fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::ASPFile::open_file", 
		            "open(%s) failed", filename);

  int retval = read (fd, header, sizeof(struct asp_params));
  if (retval < 0) {
    ::close (fd);
    throw Error (FailedSys, "dsp::ASPFile::open_file",
			    "read failed header");
  }

  correct_endian (header);

  retval = read (fd, block, sizeof(struct data2rcv));
  if (retval < 0) {
    ::close (fd);
    throw Error (FailedSys, "dsp::ASPFile::open_file",
                            "read failed block header");
  }

  correct_endian (block);

  return fd;
}


bool dsp::ASPFile::is_valid (const char* filename) const
{
  struct asp_params header;
  struct data2rcv block;

  int fd = open_read_header (filename, &header, &block);
  ::close (fd);

  if (header.band_dir != 1 && header.band_dir != -1) {
    if (verbose)
      cerr << "dsp::ASPFile invalid band_dir=" << header.band_dir << endl;
    return false;
  }

  if (fabs(header.ch_bw) > 512.0) {
    if (verbose)
      cerr << "dsp::ASPFile invalid ch_bw=" << header.ch_bw << endl;
    return false;
  }
 
  if (fabs(header.rf) > 12e4) {
    if (verbose)
      cerr << "dsp::ASPFile invalid rf=" << header.rf << endl;
    return false;
  }

  if (header.n_ds<1 || header.n_ds>4) {
    if (verbose)
      cerr << "dsp::ASPFile invalid n_ds=" << header.n_ds << endl;
    return false;
  }

  if (header.n_chan<1 || header.n_chan>32) {
    if (verbose)
      cerr << "dsp::ASPFile invalid n_chan=" << header.n_chan << endl;
    return false;
  }

  return true;
}

void dsp::ASPFile::open_file (const char* filename)
{
  struct asp_params header;
  struct data2rcv block;

  if (verbose) {
    cerr << "sizeof(struct asp_params) = " << sizeof(struct asp_params) << endl;
    cerr << "sizeof(struct data2rcv) = " << sizeof(struct data2rcv) << endl;
  }
  
  fd = open_read_header (filename, &header, &block);
 
  if (verbose) {
    cerr << "n_ds = " << header.n_ds << endl;
    cerr << "n_chan = " << header.n_chan << endl;
    cerr << "i_chan = " << block.FreqChanNo << endl;
  }

  get_info()->set_nbit (8);

  double bw = header.ch_bw * header.band_dir;
  get_info()->set_bandwidth (bw);
  get_info()->set_centre_frequency (header.rf + (block.FreqChanNo + 0.5) * bw);
 
  if (verbose) {
    cerr << "cf = " << get_info()->get_centre_frequency() << endl;
    cerr << "bw = " << bw << endl;
  }

  get_info()->set_npol(2);
  get_info()->set_state (Signal::Analytic);
  get_info()->set_rate ( header.ch_bw * 1e6 );

  MJD epoch ((int)block.iMJD, block.fMJD);
  epoch += block.ipts1 / get_info()->get_rate();

  get_info()->set_start_time( epoch );
  if (verbose) {
    cerr << "MJD = " << get_info()->get_start_time() << endl;
    cerr << "telescope = " << header.telescope << endl;
  }
  get_info()->set_telescope (header.telescope);

  get_info()->set_source (header.psr_name);

  header_bytes = sizeof(struct asp_params);

  if (verbose) {
    cerr << "totalsize=" << block.totalsize << endl;
    cerr << "NPtsSend=" << block.NPtsSend << endl;
    cerr << "overlap=" << header.overlap << endl;
    cerr << "n_samp_dump=" << header.n_samp_dump << endl;
  }

  block_header_bytes = sizeof(struct data2rcv);
  //block_tailer_bytes = header.overlap * 4;
  block_bytes = block.totalsize + block_header_bytes;

  set_total_samples();

  string prefix="asp";

  get_info()->set_mode(header.pol_mode);
  get_info()->set_machine("ASP");	
}

void dsp::ASPFile::skip_extra ()
{
  if (lseek (fd, block_tailer_bytes, SEEK_CUR) < 0)
    throw Error (FailedSys, "dsp::ASPFile::skip_extra", "seek(%d)", fd);

  struct data2rcv block;
  if (read (fd, &block, sizeof(struct data2rcv)) < 0)
    throw Error (FailedSys, "dsp::ASPFile::skip_extra", "read");

  // correct_endian (block);
  // cerr << "i_chan = " << block.FreqChanNo << endl;
  // cerr << "ipts1 = " << block.ipts1 << " ipts2=" << block.ipts2 << endl;
}

