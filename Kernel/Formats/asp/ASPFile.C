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

dsp::ASPFile::ASPFile (const char* filename,const char* headername)
  : BlockFile ("ASP")
{
}

dsp::ASPFile::~ASPFile ( )
{
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

  retval = read (fd, block, sizeof(struct data2rcv));
  if (retval < 0) {
    ::close (fd);
    throw Error (FailedSys, "dsp::ASPFile::open_file",
                            "read failed block header");
  }

  return fd;
}


bool dsp::ASPFile::is_valid (const char* filename, int) const
{
  struct asp_params header;
  struct data2rcv block;

  int fd = open_read_header (filename, &header, &block);
  ::close (fd);

  return true;
}

void dsp::ASPFile::open_file (const char* filename)
{
  struct asp_params header;
  struct data2rcv block;
  
  fd = open_read_header (filename, &header, &block);
 
  cerr << "n_ds = " << header.n_ds << endl;
  cerr << "n_chan = " << header.n_chan << endl;
  cerr << "i_chan = " << block.FreqChanNo << endl;

  info.set_nbit (8);

  double bw = header.ch_bw * header.band_dir;
  info.set_bandwidth (bw);
  info.set_centre_frequency (header.rf + (block.FreqChanNo + 0.5) * bw);
 
  cerr << "cf = " << info.get_centre_frequency() << endl;
  cerr << "bw = " << bw << endl;

  info.set_npol(2);
  info.set_state (Signal::Analytic);
  info.set_rate ( header.ch_bw * 1e6 );

  MJD epoch ((int)block.iMJD, block.fMJD);
  epoch += block.ipts1 / info.get_rate();

  info.set_start_time( epoch );
  cerr << "MJD = " << info.get_start_time() << endl;
  cerr << "telescope = " << header.telescope << endl;
  info.set_telescope_code (header.telescope[0]);

  info.set_source (header.psr_name);

  header_bytes = sizeof(struct asp_params);

  cerr << "totalsize=" << block.totalsize << endl;
  cerr << "NPtsSend=" << block.NPtsSend << endl;
  cerr << "overlap=" << header.overlap << endl;
  cerr << "n_samp_dump=" << header.n_samp_dump << endl;

  block_header_bytes = sizeof(struct data2rcv);
  //block_tailer_bytes = header.overlap * 4;
  block_bytes = block.totalsize + block_header_bytes;

  set_total_samples();

  info.set_default_basis ();
	
  string prefix="asp";

  info.set_identifier(prefix+info.get_default_id() );
  info.set_mode(header.pol_mode);
  info.set_machine("ASP");	
}

void dsp::ASPFile::skip_extra ()
{
  if (lseek (fd, block_tailer_bytes, SEEK_CUR) < 0)
    throw Error (FailedSys, "dsp::ASPFile::skip_extra", "seek(%d)", fd);

  struct data2rcv block;
  if (read (fd, &block, sizeof(struct data2rcv)) < 0)
    throw Error (FailedSys, "dsp::ASPFile::skip_extra", "read");

  // cerr << "i_chan = " << block.FreqChanNo << endl;
  // cerr << "ipts1 = " << block.ipts1 << " ipts2=" << block.ipts2 << endl;
}

