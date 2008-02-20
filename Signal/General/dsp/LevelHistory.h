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

namespace dsp
{
  class HistUnpacker;

  //! Logs the history of digitized data statistics
  class LevelHistory : public Reference::Able
  {
    
  public:

    //! Set the HistUnpacker from which histogram will be recorded
    virtual void set_unpacker (HistUnpacker*) = 0;

    //! Log the statistics of the digitized data in some form
    virtual void log_stats (std::vector<double>& mean, 
                            std::vector<double>& variance) = 0;
    
  };

}

#endif
