//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2001 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/rdisk.h,v $
   $Revision: 1.5 $
   $Date: 2006/07/09 13:27:06 $
   $Author: wvanstra $ */

#ifndef __RDISK_H
#define __RDISK_H

#include <iostream>
#include <string>
#include <vector>

#include "psr_cpp.h"

class rdisk {

 public:
  string machine;
  string path;

  static string rsh;

  rdisk () {};
  rdisk (const string& parse) { load (parse); }

  void load (const string& parse);

  // returns the available space in bytes
  double space () const;

  static void load (vector<rdisk>& disks, const char* filename);

};

inline ostream& operator << (ostream& ostr, const rdisk& rd)
{ return ostr << rd.machine << ":" << rd.path; }

#endif
