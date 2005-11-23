//-*-C++-*-

#ifndef __Mark5Unpacker_h
#define __Mark5Unpacker_h

#include "dsp/HistUnpacker.h"
// #include "dsp/Mark5File.h"
#include "dsp/File.h"
#include "Mark5File.h"

// modelled on CPSR2_8bitunpacker.h

namespace dsp {
	//! Simple 2-bit to float upacker for Mark5 VLBA files
	class Mark5Unpacker : public HistUnpacker {
	
	public:
	
	//! Constructor
	Mark5Unpacker (const char* name = "Mark5Unpacker");
	
	protected:
	
	//! The unpacking routine
	virtual void unpack ();
	
	//! Return true if we can convert the obsservation
	virtual bool matches (const Observation* observation);
	
	int find_next_frame64(int samples,const unsigned long long* buf);
	
	};


}

#endif // !defined(__Mark5Unpacker_h)
