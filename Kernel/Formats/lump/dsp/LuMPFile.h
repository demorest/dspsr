//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/



#ifndef __LuMPFile_h
#define __LuMPFile_h

#include "dsp/File.h"
#include "dsp/LuMPObservation.h"

namespace dsp {

  //! Loads BitSeries data from a LuMP data file
  class LuMPFile : public File 
  {

  public:

    //! Construct and open file
    LuMPFile (const char* filename=0);

    //! Returns true if filename appears to name a valid LuMP file
    bool is_valid (const char* filename) const;

    //! Get the information about the data source
    virtual Observation* get_info () { return lump_info; }

    //! Get the information about the data source
    virtual const Observation* get_info () const { return lump_info; }

  protected:

    friend class LuMPUnpacker;

    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the LuMP ascii header from filename
    static std::string get_header (const char* filename);

    //! Observation with extra attributes required by LuMP format
    Reference::To<LuMPObservation> lump_info;
  };

}

#endif // !defined(__LuMPFile_h)
  
