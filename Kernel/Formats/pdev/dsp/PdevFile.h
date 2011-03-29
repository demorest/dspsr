//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PdevFile_h
#define __PdevFile_h

#include "dsp/File.h"

namespace dsp {

  //! PdevFile reads baseband data from the Mock Spectrometers at Arecibo
  /*! Data in pdev format from the Mocks comes in a multi-file format
   *  where only the first file in the set contains a 1024-byte header.
   *  The rest of the file, and all subsequent files, contains 8-bit 
   *  complex dual pol single-channel samples.  These are in the same 
   *  format as is handled by ASPUnpacker.  
   *
   *  Since the pdev header does not contain a full description of the
   *  observation, we use an ASCIIObservation style header.  dspsr should
   *  be pointed at the ASCII header file (not the data files).  The
   *  DATAFILE keyword gives the base file name for the set of files.
   *  In a given set, file 0 must be present.  DATAFILE should 
   *  contain only the 'base' of the file names, for example:
   *
   *    p2613.20110323.b0s1g0.00000.pdev
   *
   *  The correct base name is 'p2613.20110323.b0s1g0'.
   *
   *  The ASCII file must have INSTRUMENT set to 'Mock' for this
   *  class to recognize the file.
   */
  class PdevFile : public File
  {
  public:
	  
    //! Construct and open file	  
    PdevFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~PdevFile ();
	  
    //! Returns true if filename is a valid pdev file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

  private:

    //! Raw header
    unsigned int rawhdr[256];

    //! Base file name
    char basename[256];

    //! Current file index
    int curfile;

  };
}

#endif // !defined(__PdevFile_h)
