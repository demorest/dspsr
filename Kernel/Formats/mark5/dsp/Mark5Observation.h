/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
//-C++-*-

// modelled on CPSR2_8bitObservation.h

#ifndef __Mark5Observation_h
#define __Mark5Observation_h

#include "dsp/Observation.h"




namespace dsp {

	class Mark5_Observation : public Observation {
	
	public:
	
	//! Construct from a Mark5 header block
	Mark5_Observation(const char* header=0);
		// can contain the filename of the header file
		// for the time being we could just read in all the
		// stuff we want from an accompanying text file

	
	//! Number of bytes offset from the beginning of the aquistion
	uint64_t offset_bytes;
	
	int get_fanout();
	
	protected:
		int fanout;

	};

}

#endif
