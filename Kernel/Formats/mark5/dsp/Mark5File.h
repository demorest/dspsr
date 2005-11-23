//-*-C++-*-

#ifndef __Mark5File_h
#define __Mark5File_h

// #include "dsp/File.h"
#include "dsp/BlockFile.h"

void extractnibbles64(const unsigned long long *data, int track, int numnibbles,
	char *nibbles);

void extractnibbles32(const unsigned int *data, int track, int numnibbles,
	char *nibbles);


// static int findfirstframe(unsigned long long *data, int samples);

namespace dsp {

	//! Loads BitSeries data from a MkV file
	class Mark5File : public BlockFile
	{
	  public:
	  
	  	//! Construct and open file	  
	  	Mark5File(const char* filename=0,const char* headername=0);
	  
	  	//! Destructor
		~Mark5File ();
	  
	  	//! Returns true if filename is a valid Mk5 file
	  	bool is_valid(const char* filename, int NOT_USED=-1) const;

	
	  	void set_framenum(int myframeno);
		int get_framenum();
		void set_payloadoffset(int mypayloadoffset);
		int get_payloadoffset();
		unsigned long long* get_modbits64();
		unsigned int* get_modbits32();
		int get_payloadsize();
		int* get_basebits();
		int get_fanout();
		int get_framesize();
		int get_first_frameNo ();
		int get_bitsperword();
		
	  protected:
	  
		//! Open the file
		virtual void open_file (const char* filename);
		
		//! Returns the MJD data of the next header
		MJD decode_date(uint64 from = 0);

		int findfirstframe64(int samples);
		int findfirstframe32(int samples);
		
		//! Defines the number of data channels
		unsigned int channels;

		unsigned long long *modbits64;
		unsigned int *modbits32;	
		unsigned long long *framebuf64;
		unsigned int *framebuf32;
		unsigned long long *payload64;
		unsigned int *payload32;
		int payloadoffset;
		int basebit[8];   // de-encoding bits 	
				//  Allow for up to 8 channels = 2 polns x 4 lots of 16 MHz
		int fileoffset;
		int tracks;
		int fanout;
		int framenum; 
		static const int FRAMESIZE  =20160;  /* dwords from entire frame of tracks */
		static const int PAYLOADSIZE=20000;  /* dwords of payload per frame */  
	
		int bitsperword;
	
	
	// find the MJD in the header of the first frame
	// and then backtrack to find the MJD of the first sample	
		MJD Mark5_stream_frame_time64(double offset);
		MJD Mark5_stream_frame_time32(double offset);
	
	// Now override the functions in BlockFile so that they behave the way
	// I expect (i.e. they work)
		
		/* Return ndat given the file and header sizes, nchan, npol and ndim.
		Requires 'info' parameters nchan, npol, ndim and header_bytes */
		
		// virtual int64 fstat_file_ndat(uint64 tailer_bytes=0);
		
		/* Load nbytes worth of *sampled data* from the device into buffer
		This procedure filters out the header data */
		 
		// virtual int64 load_bytes (unsigned char* buffer, uint64 nbytes);
		
		
		/* Set the file pointer to the absolute number of sampled *data* bytes */
		
		// virtual int64 seek_bytes (uint64 bytes);
		
	   private:
	   	uint64 curr_block_byte;	

	};

}
#endif // !defined(__Mark5File_h)
