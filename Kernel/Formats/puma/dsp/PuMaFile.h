//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/puma/dsp/PuMaFile.h,v $
   $Revision: 1.1 $
   $Date: 2003/04/23 20:08:54 $
   $Author: wvanstra $ */


#ifndef __PuMaFile_h
#define __PuMaFile_h

#include "dsp/File.h"
#include "pumadata.h"

namespace dsp {

  //! Loads TimeSeries data from a PuMa data file.
  /*! This class is heavily copied from PuMaData by Russell Edwards. */
  class PuMaFile : public File 
  {
  public:
   
    //! Construct and open file
    PuMaFile (const char* filename = 0);

    //! Returns true if filename appears to name a valid PuMa file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Set the Input::info attribute, given a loaded Header_type object
    void set_info (const Header_type& hdr);

  private:

    //! The PuMa header
    Header_type hdr;

#if FUTURE_WORK

    /* In the future, a single PuMaFile may act like the current PuMaData
       class developed by Russell Edwards.  ie. it will handle multiple
       time divisions and multiple bands.  For now, I think it will be best
       to use the MultiFile class in order to join contiguous time divisions.
    */

    //! Filename prefix
    string fprefix;

    //! Number of bands and starting band
    int nbands, ibandstart;

    //! Number of time divisions and starting division
    int ntimediv, itimedivstart;

    //! File descriptors for each band of the current time division
    vector<FILE*> band_fptr;

    //! Offset (in bytes) to first sample in each of the above files
    vector<int> band_samplestart_offset;

    //! Number of frequency channels per band
    int nchan_band;

    //! 
    unsigned long current_offset;

    //! 
    int current_div;

    //! Number of samples in each time div
    vector<unsigned> offset_bydiv;

    
    //! Produce a string of the form ".00001.1.puma"
    string make_fname_extension (int itimediv, int iband);


#endif

  };

}

#endif // !defined(__PuMaFile_h)
  
