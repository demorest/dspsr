#ifndef __FPTM_OBS_H
#define __FPTM_OBS_H

#include <stdio.h>
#include <vector>
#include <string>

#include "MJD.h"
#include "angle.h"

#define FPTM_OBS_LOG "runtime/observe/fptm_obs.log"

/* Advertised format
   MJD_start OBSTYPE SOURCENAME RA DEC Pfold Tobs(s)
*/

class fptm_obs {

 public:
  static string logfilename;

  MJD start;
  int obstype;
  double duration;
  string source;

  AnglePair coordinates;
  double period;

  fptm_obs ();
  fptm_obs (string* parse);
  fptm_obs (const MJD& start, double duration, string source, int obstype=0);

  bool overlaps (const MJD& tstart, const MJD& tend);

  int load (string* parse, int original=0);
  int unload (FILE* out, int original=0);

};

int relevant (vector<fptm_obs>* observations, const MJD& mjd, double duration);

#endif
