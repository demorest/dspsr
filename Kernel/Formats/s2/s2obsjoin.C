/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "s2obsjoin.h"

#include <stdio.h>
#include "tci_file.h"

using namespace std;

//#define DEBUG 1

bool ObsJoin::s2File::load(const string& fname_)
{
  fname = fname_;

  // Methods that are needed to extract two things
  // Start time - as MJD
  // Duration in seconds
  //      Some day this should be made easier.....
  // cwest

  int nbits;
  int npols = 2;
  double rate;
  float nbyte;

  tci_fd s2file;
  tci_hdr header;

#ifdef DEBUG
  fprintf(stderr, "Filename to load: \"%s\"\n", fname.c_str());
#endif
  if (tci_file_open (fname.c_str(), &s2file, &header, 'r') != 0){
    fprintf(stderr, "ObsJoin:s2File::load - could not construct from %s\n",
	    fname.c_str());
    return false;
  }

  utc_t utc;
  str2utc (&utc, header.hdr_time);
  startMJD = MJD(utc);

  char* bitspersample = strrchr (header.hdr_s2mode, '-');
  if(!bitspersample){
    cerr << "ObsJoin:s2File::load - trouble finding bits/sample in "
	 << header.hdr_s2mode << "\n\t\tSetting to 1 bit/sample" << endl;
    nbits = 1;
  }else{
    nbits = atoi(bitspersample+1);
  }


  nbyte = float(nbits*npols) / 8.0;
  rate = double (s2file.data_rate) * 2.0 / nbyte;
  duration = ((s2file.fsz * 16.0) / (nbits*npols)) / rate;

#ifdef DEBUG
  fprintf(stderr, "--------- START DEBUG of s2obsjoin_lib.C ------\n");
  fprintf(stderr, "NByte    : %f\n", nbyte);
  fprintf(stderr, "NBits    : %d\n", nbits);
  fprintf(stderr, "NPols    : %d\n", npols);
  
  fprintf(stderr, "Rate     : %lf\n", rate);
  fprintf(stderr, "s2Rate   : %d\n", s2file.data_rate);
  fprintf(stderr, "s2fsz    : %ld\n", s2file.fsz);
  
  fprintf(stderr, "s2obsjoin: StartMJD = %s\n", startMJD.printall());
  fprintf(stderr, "s2obsjoin: Duration = %lf\n", duration);
  fprintf(stderr, "---------  END  DEBUG of s2obsjoin_lib.C ------\n");
#endif
  
  tci_file_close(s2file);
  
  return true;
}

