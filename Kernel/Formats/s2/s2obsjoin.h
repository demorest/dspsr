/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// This is a bunch of S2-specific stuff to go with the ObsJoin
// stuff, see genutil++/obsjoin.{C,h}

#include "obsjoin.h"

namespace ObsJoin
{
  class s2File : public LoadableSegment
  {
  public:
    s2File() {}
    virtual bool load(const std::string& fname_); // return success status
  };
  typedef LoadableSegments<s2File> s2Files;
}

