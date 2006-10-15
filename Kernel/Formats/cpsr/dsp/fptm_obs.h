/***************************************************************************
 *
 *   Copyright (C) 1999 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __FPTM_OBS_H
#define __FPTM_OBS_H

#include <stdio.h>
#include <vector>
#include <string>

#include "MJD.h"
#include "Angle.h"

#define FPTM_OBS_LOG "runtime/observe/fptm_obs.log"

/* Advertised format
   MJD_start OBSTYPE SOURCENAME RA DEC Pfold Tobs(s)
*/

class fptm_obs {

 public:
  static std::string logfilename;

  MJD start;
  int obstype;
  double duration;
  std::string source;

  AnglePair coordinates;
  double period;

  fptm_obs ();
  fptm_obs (std::string* parse);
  fptm_obs (const MJD& start, double duration, std::string source,
	    int obstype=0);

  bool overlaps (const MJD& tstart, const MJD& tend);

  int load (std::string* parse, int original=0);
  int unload (FILE* out, int original=0);

};

int relevant (std::vector<fptm_obs>* observations,
	      const MJD& mjd, double duration);

#endif
