/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef _DATA2RCV_H
#define _DATA2RCV_H

struct data2rcv {
 int32_t totalsize;
 int32_t NPtsSend;
 double iMJD;
 double fMJD;
 int64_t ipts1,ipts2; /* Actual position of the start and end in the data time serie */
 int32_t  FreqChanNo;
} __attribute__ ((aligned (4), packed)); // Align as in 32-bit

#endif
