
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

