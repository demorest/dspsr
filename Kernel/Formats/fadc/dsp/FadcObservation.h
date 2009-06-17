//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FadcObservation_h
#define __FadcObservation_h

#include "dsp/Observation.h"

namespace dsp {
 
  // NOTES : The comment below may have to change
  //         uint64_t offset_bytes can probably be deleted

  //! General means of constructing Observation attributes from FADC data
  /*! This class parses the ASCII header block used for FADC data and
    initializes all of the attributes of the Observation base class.
    The header comes from the header file Exp... , not from the data files
    of the measurement itself. The measurement should be located in the 
    Data subdirectory */
    
  class FadcObservation : public Observation {

  public:

    //! Construct from a Fadc ASCII header
    // This constructor is empty
    // It does not ever get called
    FadcObservation (const char* header=0);

    // constructor used by FadcFile
    FadcObservation (const char* header, long firstFile, long lastFile, unsigned long offset_tsmps_file0, unsigned long offset_tsmps, double centerFreqOverride=0, double bwOverride=0);
    
  protected:
  
  // this function initializes offset_tsmp with the number of time samples that are missing
  // or incomplete at the beginning of the requested data (first file). These samples will not be
  // loaded and have to be considered when calculating the total number of samples.
//  int get_offset_tsmps(char* expFileName, long firstFile, long lastFile, unsigned long* offset_tsmps_file0, unsigned long* offset_tsmps, int nbit);

  int read_blockMap(long *buffers_per_file, long *bytes_per_buffer);
    
  };
  
}

#endif
