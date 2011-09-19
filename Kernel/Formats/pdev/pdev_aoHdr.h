/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef _PDEV_AOHDR_H
#define _PDEV_AOHDR_H

// aoHdr part of pdev baseband files
// copied from Phil's web info at:
// http://www.naic.edu/~phil/software/datataking/pdevfile/pdevfile.html
// on 2011/09/18
// TODO: make sure alignment is ok.
struct pdev_aoHdr {
  char hdrVer[4];  // no null-term
  uint32_t bandIncrFreq;
  double cfrHz;
  double bandWdHz;
  char object[16]; // no null-term
  char frontEnd[8]; // no null
  double raJDeg;
  double decJDeg;
  double azDeg;
  double zaDeg;
  int32_t imjd;
  int32_t isec;
};

#endif
