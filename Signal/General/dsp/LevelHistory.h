/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __LevelHistory_h
#define __LevelHistory_h

#include "ReferenceAble.h"
#include <vector>

namespace dsp {

  //! History of digitized data statistics
  /*!
    This abstract base class defines the interface to classes
    that can log and/or plot the statistics of digitized data.
  */
  
  class HistUnpacker;

  class LevelHistory : public Reference::Able
  {
    
  public:
    
    //! Log the statistics of the digitized data in some form
    virtual void log_stats (std::vector<double>& mean, 
                            std::vector<double>& variance,
			    HistUnpacker* stats) = 0;
    
  };

}

#endif
