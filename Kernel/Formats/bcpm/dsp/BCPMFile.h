//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_BCPMFile_h
#define __dsp_BCPMFile_h

#include "dsp/bpphdr.h"
#include "dsp/File.h"

namespace dsp {

  //! Loads BitSeries data from a BCPM data file.
  class BCPMFile : public File 
  {

  public:
   
    //! Construct and open file
    BCPMFile (const char* filename = 0);

    //! Destructor
    ~BCPMFile ();

    //! Returns true if filename appears to name a valid BCPM file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Switches the endianess of relevant variables, if need be
    void switch_endianess();
    
    //! Pulled out of sigproc
    std::vector<int> bpp_chans(double bw, int mb_start_addr, int mb_end_addr, int mb_start_brd, int mb_end_brd, int *cb_id, double *aib_los, float *dfb_sram_freqs, double rf_lo,double& centre_frequency);

    //! Pads gaps in data.  Untested
    virtual int64 pad_bytes(unsigned char* buffer, int64 bytes);
    
    //! Stores the search header
    BPP_SEARCH_HEADER bpp_search;

  };

}

#endif // !defined(__BCPMFile_h)
  
