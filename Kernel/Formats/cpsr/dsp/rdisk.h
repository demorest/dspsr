//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/rdisk.h,v $
   $Revision: 1.1 $
   $Date: 2001/03/05 03:28:05 $
   $Author: wvanstra $ */

#ifndef __RDISK_H
#define __RDISK_H

#include <string>

class rdisk {

 public:
  string machine;
  string path;

  rdisk (const string& parse) { load (parse); }

  void load (const string& parse);

  // returns the available space in bytes
  double space ();

};

#endif
