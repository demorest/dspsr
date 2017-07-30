//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FadcFile_h
#define __FadcFile_h

#include "dsp/File.h"

struct two_bit_out{
  unsigned lRe  :2;
  unsigned lIm   :2;
  unsigned rRe  :2;
  unsigned rIm   :2;
};

struct four_bit_out{
  unsigned Re  :4;
  unsigned Im   :4;
};


namespace dsp {

	//! Loads BitSeries data from a Fadc file
	class FadcFile : public File
	{
	  public:
	  
	  	//! Construct and open file	  
	  	FadcFile(const char* filename=0);
	  
	  	//! Destructor
		~FadcFile ();
	  
	  	//! Returns true if filename is a valid Fadc file
	  	bool is_valid(const char* filename) const;

		
	  protected:
	  
		//! Open the file
		virtual void open_file (const char* filename);
		
		//! Read the Fadc header
		static std::string get_header (const char* filename);
		
		// also switches sign of the imaginary values
		void writeByte(FILE* outfile, two_bit_out two);
		void writeByte(FILE* outfile, four_bit_out four);

		int createDataFile(char* expFileName, long firstFile, long lastFile, long* offset_tsmps_file0, long* offset_tsmps, int nbit, int nPol, int nChan, int nADCperCard, long buffers_per_file, long bytes_per_buffer, int expect_magic_code);
		// offest_tsmps refers to the number of time samples from the beginning of the file (0 and firstFile) where the
		// usable data starts (i.e. complete time samples that contain measurement data)
		
		int read_blockMap(long *buffers_per_file, long *bytes_per_buffer);
		
		bool fileExists (char* fileName);
		uint64_t fileSize(char* fileName);
	};

}
#endif // !defined(__FadcFile_h)
