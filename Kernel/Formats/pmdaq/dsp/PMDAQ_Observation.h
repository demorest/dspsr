/***************************************************************************
 *
 *   Copyright (C) 2002 by Matthew Bailes
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __PMDAQ_Observation_h
#define __PMDAQ_Observation_h

#include "dsp/Observation.h"
#include "tostring.h"

namespace dsp {
 
  //! General means of constructing Observation attributes from PMDAQ data
  /*! This class parses the header block used for PMDAQ data and
    initializes all of the attributes of the Observation base class.
    The header block may come from a PMDAQ header file.
  */

  class PMDAQ_Observation : public Observation {

  public:

    //! Construct from a CPSR2 ASCII header block
    PMDAQ_Observation (const char* header);

    //! Returns true if two filters were used.  e.g. both 10cm and 50cm bands
    bool has_two_filters(){ return freq2_channels > 0; }

    //! Inquire the second centre frequency
    double get_second_centre_frequency(){ return second_centre_frequency; }

    //! Inquire the second bandwidth
    double get_second_bandwidth(){ return second_bandwidth; }

    //! Inquire the number of channels at the first centre frequency
    unsigned get_freq1_channels(){ return freq1_channels; }

    //! Inquire the number of channels at the second centre frequency
    unsigned get_freq2_channels(){ return freq2_channels; }

  protected:

    //! If two bands are stored in one file this stores the second centre frequency
    double second_centre_frequency;

    //! If two bands are stored in one file this stores the second bandwidth
    double second_bandwidth;

    //! Number of channels at the first centre frequency (including dummies)
    unsigned freq1_channels;

    //! Number of channels at the second centre frequency (including dummies)
    unsigned freq2_channels;

    //! Parses the C string
    template<class T>
    T read_header(const char* header,unsigned startchar, unsigned nchars);
    
  };
  
}

//! Parses the C string
template<class T>
T dsp::PMDAQ_Observation::read_header(const char* header,unsigned startchar, unsigned nchars){
  std::string ss (header+startchar,header+startchar+nchars);
  return fromstring<T>(ss);
}

#endif
