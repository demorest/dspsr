//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr/dsp/xfer_tape.h,v $
   $Revision: 1.1 $
   $Date: 2001/07/30 02:14:54 $
   $Author: wvanstra $ */

#ifndef __XFER_TAPE_H
#define __XFER_TAPE_H

#include <vector>

class SystemTime;
class rdisk;

int xfer_tape (const char* tapedev, vector<int>* filenos,
	       const vector<rdisk>& disks, char* ext = NULL, 
	       char* check_fptm = NULL, double leave_alone = 600e6, // 600MB
	       SystemTime* clock=NULL, int obstype=-1, bool keep=true);


#endif
