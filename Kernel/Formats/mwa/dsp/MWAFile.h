//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_mwafile_h_
#define __dsp_mwafile_h_

#include <string>
#include <vector>

#include "MJD.h"

#include "dsp/File.h"

namespace dsp {

  class MWAFile : public File {
  public:
    
    //! Construct and open file
    MWAFile (const char* filename = 0);

    //! Virtual destructor
    virtual ~MWAFile();

    //! Returns true if filename appears to name a valid MWA file
    bool is_valid (const char* filename) const;
    
  protected:

    //! Open the file
    virtual void open_file (const char* filename);

  };

}





#endif
