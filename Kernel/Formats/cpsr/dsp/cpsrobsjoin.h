/***************************************************************************
 *
 *   Copyright (C) 2000 by Russell Edwards
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// This is a bunch of CPSR-specific stuff to go with the ObsJoin
// stuff, see genutil++/obsjoin.{C,h}

#include "obsjoin.h"

namespace ObsJoin
{
  class CPSRFile : public LoadableSegment
  {
  public:
    CPSRFile() {}
    virtual bool load(const std::string& fname_); // return success status
  };
  typedef LoadableSegments<CPSRFile> CPSRFiles;
}

