//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2001 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr/dsp/pspmXfer.h

#ifndef __pspmXfer_h
#define __pspmXfer_h

#include "pspmDbase.h"
#include "RealTimer.h"
#include "rdisk.h"

class pspmXfer {

  pspmDbase::server hdrinfo;  // database with PSPM header info

  int filenum;        // current file number transfered (just a counter)
  int rfsindex;       // current index in rfs
  int nfsindex;       // current index in nfs

  bool rfs_ready (double reqd_space);

 public:

  bool exit;          // flag to exit from xfer() early
 
  std::vector<rdisk>  rfs; // remote file systems (pipe through rsh)
  std::vector<std::string> nfs; // local or NFS-mounted file systems

  std::string tape;        // device name of tape
  std::string ext;         // extension to place on trigger file
  std::string xref;        // if pspm.name == xref, then check FPTM log
  bool   only_xref;   // only transfer file if it can be xrefd

  double leave_alone; // number of bytes to leave vacant on file systems

  RealTimer slept;    // time spent sleeping
  RealTimer xfert;    // time spent xfering

  // special mode for xfering only one type of file. only works when xref
  int  obstype;       // 0=pulsar 1=cal
  // same like above, but for a source name
  std::string source;
  // same as above, for xfering only hydracal
  bool hydra;

  // wether to keep or leave observations above, if specified
  bool keep;

  pspmXfer ();

  // transfer a list of files (numbered from 1 to 32) from tape to disk.
  // if tape is not specified, the first file is read from tape in order
  // to determine the tape number from the header.
  // then selects a subset, using ::select, before calling ::xfer
  int xfer (std::vector<int>& filenos, int tape=-1);

  // select a subset of filenos, based on criteria that are good
  std::vector<pspmDbase::entry> select (int tapenum, const std::vector<int>& filenos);

  // transfer a list of observations from tape to disk
  int xfer (std::vector<pspmDbase::entry>& obs);

  // reads the first header off of tape and returns the tape number
  int tapenum ();

};

#endif // ! __pspmXfer_h
