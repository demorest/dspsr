//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Multiplex.h,v $
   $Revision: 1.1 $
   $Date: 2009/09/10 01:02:34 $
   $Author: tcaotiaafoc $ */


#ifndef __Multiplex_h
#define __Multiplex_h

#include "dsp/MultiFile.h"

namespace dsp {

  //! Loads BitSeries data from multiple files
  class Multiplex : public MultiFile {

  public:
  
    //! Constructor
    Multiplex ();
    
    //! Destructor
    virtual ~Multiplex ();

    //! Returns true if filename is an ASCII file listing valid filenames
    virtual bool is_valid (const char* filename) const;
    
    //! Open a number of files and treat them as one logical observation.
    virtual void open (const std::vector<std::string>& new_filenames);

  protected:
    
    //! Open the ASCII file of filenames
    virtual void open_file (const char* filename);

    //! Load bytes from file
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t bytes);
    
    //! Adjust the file pointer
    virtual int64_t seek_bytes (uint64_t bytes);


  };

}

#endif // !defined(__Multiplex_h)
  
