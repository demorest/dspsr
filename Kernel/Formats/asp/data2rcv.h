
#ifndef _DATA2RCV_H
#define _DATA2RCV_H

struct data2rcv {
 int totalsize;
 int NPtsSend;
 double iMJD;
 double fMJD;
 long long ipts1,ipts2; /* Actual position of the start and end in the data time serie */
 int  FreqChanNo;
};

#endif
