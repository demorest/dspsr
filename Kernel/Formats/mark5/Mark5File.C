#include "dsp/Mark5File.h"
#include <iomanip>
#include "Error.h"

#include <time.h>
#include <errno.h>
#include <stdio.h>

#include "coord.h"
#include "string.h"	
#include "string_utils.h"
#include "ascii_header.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "fcntl.h"


void extractnibbles64(const unsigned long long *data, int track, int numnibbles,
			char *nibbles)
{
	unsigned long long mask;
	int i, b;
	
	mask = 1 << track;
	
	for (i = 0;  i < numnibbles; i++)  {
		nibbles[i] = 0;	
		for (b = 0; b< 4; b++) 
			nibbles[i] += (data[4*i+3-b] & mask) ? (1<<b) : 0;
	
	}

}			

void extractnibbles32(const unsigned int *data, int track, int numnibbles,
			char *nibbles)
{
	unsigned int mask;
	int i,b;
	const unsigned short *dat;
	
	dat = (unsigned short *)data;
	
	mask = 1 << track;
	
	for (i=0; i < numnibbles; i++)
	{
		nibbles[i]=0;
		for (b=0 ; b<4 ; b++)
			nibbles[i] += (dat[4*i+3-b] & mask) ? (1<<b) : 0; 
	}
}


int dsp::Mark5File::findfirstframe64(int samples)
{ 
	int i,j;
	unsigned int *dat;
	
	dat = (unsigned int *)framebuf64;
	
	for (i = 2; i < 2*samples-64; i++) {
		if (dat[i-1] || dat[i-2]) continue;
		for (j = 0; j < 64; j++) if (dat[i+j] != 0xFFFFFFFF) break;
		if (j == 64) return i;
	}
	return -1;
}

int dsp::Mark5File::findfirstframe32(int samples)
{ 
	int i,j;
	unsigned int *dat;
	
	dat = (unsigned int *)framebuf32;
	
	for (i = 1; i < samples-32; i++) {
		if (dat[i-1]) continue;
		for (j = 0; j < 32; j++) if (dat[i+j] != 0xFFFFFFFF) break;
		if (j == 32) return i;
	}
	return -1;
}


dsp::Mark5File::Mark5File (const char* filename,const char* headername) : BlockFile ("Mark5")
{
 
}

dsp::Mark5File::~Mark5File ( )
{


}

bool dsp::Mark5File::is_valid(const char* filename, int) const
{
	int filed = ::open64(filename, O_RDONLY);
	
	if (filed == -1)
		throw Error (FailedSys, "dsp::Mark5File::open",
			"failed open64(%s)", filename);
			
	return true;
	/*		
	int chans = count_channels(filed);   
		// W - how do we count no. of channels?
	if (chans == 0)
		return false;
	else
		return true;
	*/
}

void dsp::Mark5File::open_file(const char* filename)
{	

	// FIRST Get some vital information from the header file.
	// if (headername == NULL)
	//	throw_str ("Mark5_Observation - no header");
	char headername[256];
	
	strcpy(headername,filename);
	strcat(headername,".hdr");
	char header[1024];
	FILE *ftext;
	
	if ((ftext=fopen(headername,"r")) == NULL) 
		throw Error (FailedSys,"dsp::Mark5File",
			"Cannot open header file %s",headername);
	
	fread(header,sizeof(char),1024,ftext);

	// ///////////////////////////////////////////////////////////////	
	// BITS PER WORD
	//
	  if (ascii_header_get (header, "BITSPERWORD", "%d", &bitsperword) < 0)
	    throw_str ("Mark5_Observation - failed read SOURCE");
	    
	    
					
	int bufsize = FRAMESIZE+1024;
	
	int bytesperword=bitsperword/8;
	
	// initmodbits64
	// 	if (!modbits64) initmodbits64();
	int i, n, k;
	static unsigned int ff[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

	
	fd = ::open64(filename,O_RDONLY);

	if (!fd) 
		throw Error (FailedSys, "dsp::Mark5File::open_file",
			"Failed to open Mark 5 data file %s",filename); 
 	
	long readno;

	
	int firstf;

	payloadoffset=0;
 
 	// cerr << "bits per word = " << bitsperword << endl;
 
 
	switch (bitsperword) {
	  case 64: {
	  	modbits64 = (unsigned long long*)malloc(PAYLOADSIZE*sizeof(unsigned long long));
		framebuf64 = (unsigned long long *)malloc(bytesperword*bufsize);
		// cerr << "Opening 64-bit file " << endl;
		readno = read(fd,framebuf64,bytesperword*bufsize);
		if (readno != bytesperword*bufsize) 
		throw Error (FailedSys,"dsp::Mark5File::open",
				"File too small to get %d bytes of data",bytesperword*bufsize);
		firstf = findfirstframe64(bufsize); 
		payload64 = framebuf64+96;  // a pointer to where the data actually starts.


		   }
		   break;
	  case 32: {
		modbits32 = (unsigned int *)malloc(PAYLOADSIZE*sizeof(unsigned int));
		framebuf32 = (unsigned int*)malloc(bytesperword*bufsize);
		// cerr << "Opening 32-bit file " << endl;
		readno = read(fd,framebuf32,bytesperword*bufsize);
		if (readno != bytesperword*bufsize) 
		throw Error (FailedSys,"dsp::Mark5File::open",
				"File too small to get %d bytes of data",bytesperword*bufsize);
		firstf = findfirstframe32(bufsize);
		cerr << "First frame at " << firstf << endl;
		payload32 = framebuf32+96;
		   } break;
	  default:
	 	throw Error (FailedSys,"dsp::Mark5File::open",
				"%d bits per word mode not supported",bitsperword);
	 }


		// warning: find first frame actually returns a position 64 words into the 
		//    header.
	if (firstf < 0 ) throw Error (FailedSys,"dsp::Mark5File::open",
			"No frame found in first %d samples",bufsize);
	
	fileoffset = firstf;
	fprintf(stderr,"fileoffset %d\n",fileoffset);
	framenum=0;

				   
	for(i = 0; i < PAYLOADSIZE; i++)
	{
		k = ff[10] ^ ff[12] ^ ff[13] ^ ff[15];
		for(n = 15; n > 0; n--) ff[n] = ff[n-1];
		ff[0] = k;
		if (bitsperword==64) modbits64[i] = k*0xFFFFFFFFFFFFFFFFLL;
		if (bitsperword==32) modbits32[i] = k*0xFFFFFFFF;
		if(i % 8 == 7)
		{
			k = ff[10] ^ ff[12] ^ ff[13] ^ ff[15];
			for(n = 15; n > 0; n--) ff[n] = ff[n-1];
			ff[0] = k;
		}

	}
	
				


	
	
	//
	// no idea about the size of the data
	//
	
	info.set_ndat(0);
	
	// ///////////////////////////////////////////////////////////////
	// TELESCOPE
	//
	
	char hdrstr[256];
	if (ascii_header_get (header,"TELESCOPE","%s",hdrstr) <0)
		throw_str ("Mark5_Observation - failed read TELESCOPE");
	/* user must specify a telescope whose name is recognised or the telescope
	code */
	
	string tel= hdrstr;
	if ( !strcasecmp (hdrstr, "parkes") || tel == "PKS") 
    	info.set_telescope_code (7);
  	else if ( !strcasecmp (hdrstr, "GBT") || tel == "GBT")
    	info.set_telescope_code (1);
  	else if ( !strcasecmp (hdrstr, "westerbork") || tel == "WSRT")
    	info.set_telescope_code ('i');
  	else {
    	   cerr << "Mark5File:: Warning using telescope code " << hdrstr[0] << endl;
    	info.set_telescope_code (hdrstr[0]);
  	}
	
	// ///////////////////////////////////////////////////////////////	
	// SOURCE
	//
	  if (ascii_header_get (header, "SOURCE", "%s", hdrstr) < 0)
	    throw_str ("Mark5_Observation - failed read SOURCE");

  	info.set_source (hdrstr);

	// ///////////////////////////////////////////////////////////////
	// FREQ
	//
	// Note that we assign the CENTRE frequency, not the edge of the band
	double freq;
  	if (ascii_header_get (header, "FREQ", "%lf", &freq) < 0)
    	throw_str ("Mark5File - failed read FREQ");

  	info.set_centre_frequency (freq);
	
	//
	// until otherwise, the band is centred on the centre frequency
	//
	info.set_dc_centred(true);

	// ///////////////////////////////////////////////////////////////
	// BW
	//
	double bw;
	if (ascii_header_get (header, "BW", "%lf", &bw) < 0)
    		throw_str ("Mark5File - failed read BW");

	info.set_bandwidth (bw);
	
	
	// ///////////////////////////////////////////////////////////////
	// No. of CHANNELS
	//
	// We're going to read in all four 16 MHz streams at once
	//  --- we'll generalise this later 
	
	info.set_nchan(4);	
	
	// ///////////////////////////////////////////////////////////////
	// NPOL
	//	
	//  -- generalise this later
	
	info.set_npol(2);    // read in both polns at once
	
	// ///////////////////////////////////////////////////////////////	
	// NBIT  &  RESOLUTION
	//
	
	int scan_nbit;
	if (ascii_header_get (header, "NBIT", "%d", &scan_nbit) < 0)
	   throw_str ("Mark5File - failed read NBIT");

	info.set_nbit (scan_nbit);

	Input::resolution = bitsperword/scan_nbit;  
				// instruct the loader to only take gulps in 32/16 lots of nbits
				 // necessary since Mk5 files are written in 64-/32-bit words

	// ///////////////////////////////////////////////////////////////	
	// NDIM  --- whether the data are Nyquist or Quadrature sampled
	//
	// VLBA data are Nyquist sampled

	 info.set_state (Signal::Nyquist);
	  
	
	 //
	 // call this only after setting frequency and telescope
  	 //
  	 info.set_default_basis ();





	// ///////////////////////////////////////////////////////////////
	// TSAMP   &    RATE
	//
	// Note TSAMP is the sampling period in microseconds
	double sampling_interval;
  	if (ascii_header_get (header, "TSAMP", "%lf", &sampling_interval)<0)
    		throw_str ("Mark5File - failed read TSAMP");

  	/* IMPORTANT: TSAMP is the sampling period in microseconds */
  	sampling_interval *= 1e-6;

  	info.set_rate (1.0/sampling_interval); 

	// ///////////////////////////////////////////////////////////////
	//  FANOUT
	//
	if (ascii_header_get (header,"FANOUT","%d",&fanout) < 0)
		throw_str ("Mark5File - failed read FANOUT");
	
	cerr << "Fanout is " << fanout << endl;

	// ///////////////////////////////////////////////////////////////	  
	// MJD_START
	//
	// look in the actual data file for the mjd of the first frame
	// rather than relying on user input from the header file
	 // Get start mjd 
		// the MJD we get back from VLBA data is a 3-digit number
		// use Craig's code to "correct" it --- not sure if this is necessary
	
	//double offset=sampling_interval*(4.0*(fileoffset-128)/bytesperword) * fanout;
 		// there are x=4*fileoffset/8 64-bit words to the first frame header
		// this corresponds to x * fanout * sampling interval seconds.
	
		// header always starts 64 words before fileoffset.  
		// There are fanout*(fileoffset-64) samples available.



	// NB: for both 32 and 64 bit files find_first_frame returns the position in terms
	//   of 4 byte blocks (unsigned ints)
 	// So the offset from start of file in bytes    is 4 * fileoffset
	//        offset in words                       is 4 * fileoffset/bytesperword
	//        offset from file start to header start 
	//				in words        is 4*(fileoffset/bytesperword)-64
	//        offset in samples  is [4*(fileoffset/bytesperword) - 64 ] * fanout

	double offset = sampling_interval * (4*(fileoffset/bytesperword)-64) * fanout;


		// check that the file doesn't start in the middle of a frame header.
	switch (bitsperword) {
	case 64:
		if ( ((int)(4*fileoffset/sizeof(uint64)) > PAYLOADSIZE) &&
				((int)(4*fileoffset/sizeof(uint64)) < FRAMESIZE)) {
			offset = PAYLOADSIZE*fanout*sampling_interval; 
						// the time spanned by a frame		
		}
		break;
	case 32: if ( ((int)(4*fileoffset/sizeof(unsigned int)) > PAYLOADSIZE) &&
				((int)(4*fileoffset/sizeof(unsigned int)) < FRAMESIZE)) {
			offset = PAYLOADSIZE*fanout*sampling_interval; 
						// the time spanned by a frame	 
		}
		 break;	
	default: break;
	
	}  // switch bitsperword
	
	cerr << "Offset between file start and first full frame " << offset * 1e6;
	cerr << " microseconds"<< endl;

	// feed it zero offset if we've header_bytes and will read
	// from the first full frame
	if (bitsperword==64) 
		info.set_start_time(Mark5_stream_frame_time64(0.0));
	   else {
	if (bitsperword==32)
		info.set_start_time(Mark5_stream_frame_time32(0.0));
	}


	//lseek(fd,4*fileoffset,SEEK_SET);     // file is returned to the start of the
					       // first full frame 

	lseek(fd,0,SEEK_SET);  // return the file to the start.

		// header_bytes is the offset to the start of the 
		// first header.
	header_bytes=4*fileoffset - 64*bytesperword;
	// header_bytes=0;
	cerr<< "Bytes to skip *before* first header = " << header_bytes << endl;
	lseek(fd,header_bytes,SEEK_SET);
	
	block_bytes=FRAMESIZE*bytesperword;
		// not sure here: data starts only 96 words into the frame
		// but only 20000 words of the frame are data.  What about
		// the other 64 words?
	// My guess is that find_first_frame returns the location of the 
	// end of the first 64 words of the header. So we need to advance 
	// another 160-64=96 words to get to the start of the data.
        //block_header_bytes=(FRAMESIZE-PAYLOADSIZE)*sizeof(uint64);
	
	block_header_bytes=160*bytesperword;  // is this correct even for 32-bit data??? 
	block_tailer_bytes=0; 

	fileoffset=0;  // because we're going to load from the first full frame.

	// ///////////////////////////////////////////////////////////////
	// FILENAME of the actual MkV data file
	string datafilename;   
	
	 if (ascii_header_get (header,"DATAFILE","%s",hdrstr) < 0 )
	 	throw_str ("Mark5File - failed read Mark5 data filename");
	
		
	
	// ///////////////////////////////////////////////////////////////
	// CALCULATE the various offsets and sizes
	//
	// PRIMARY  --- what's this???
	
	string prefix="tmp";    // what prefix should we assign??
	  
	info.set_identifier(prefix+info.get_default_id() );
	info.set_mode(stringprintf ("%d-bit mode",info.get_nbit() ) );
	info.set_machine("Mark5");
	
	// ///////////////////////////////////////////////////////////////
	// RA and DEC
	//
	// leave this part to Mark5Observation to read in the co-ords.
	
	
	
	
	// NOW read data which is critical to interpreting Mark 5 files.
	


	// ///////////////////////////////////////////////////////////////
	// TRACKS
	//

	if (ascii_header_get (header,"TRACKS","%d",&tracks) < 0)
		throw_str ("Mark5File - failed read TRACKS");

	cerr << "Number of channels is " << tracks << endl;

	// ///////////////////////////////////////////////////////////////
	// BASEBIT
	//
	//  Set this according to which track we are going to read.

	if (ascii_header_get (header,"BASEBIT","%d %d %d %d %d %d %d %d",
				&basebit[0],&basebit[1],&basebit[2],&basebit[3],
				&basebit[4],&basebit[5],&basebit[6],&basebit[7]) < 0)
		throw_str ("Mark5File - failed read BASEBITs");

	printf("Encoding basebits are %d %d %d %d %d %d %d %d \n",basebit[0],
		basebit[1],basebit[2],basebit[3],basebit[4],basebit[5],
		basebit[6],basebit[7]);
		
	fclose(ftext);
	fflush(stdout);         


		
}

MJD dsp::Mark5File::Mark5_stream_frame_time64(double offset)
{
  char nibs[12];
  double framemjd, framesec;
  MJD current;
  
  // g_assert(vs);    -- what does this do?
  
	lseek(fd,4 * fileoffset,SEEK_SET);
	int n = read(fd,framebuf64,8*FRAMESIZE);
	if (!n)	throw Error (FailedSys,"dsp::Mark5_stream_frame_time64",
		"Error reading next frame");



	extractnibbles64(framebuf64+32,0,12,nibs);
  
	framemjd = nibs[0]*100 + nibs[1]*10 + nibs[2];
	framesec = nibs[3]*10000 + nibs[4]*1000 + nibs[5]*100 + nibs[6]*10
                + nibs[7] + nibs[8]/10.0 + nibs[9]/100.0 + nibs[10]/1000.0
		+ nibs[11]/10000.0 ;
		
   
	printf("mjd is %0.0f and %f seconds\n",framemjd,framesec);
	printf("offset is %f seconds\n",offset);
	
	framesec -= offset;

	
	current.Construct(time(NULL));
	
	int julian = int(framemjd);  	
	if (int(current.in_days())%1000 >= julian) {
		// 2 most significant digits of 5 digit julian are correct
		julian += ( int(current.in_days())/1000)*1000;
	} else {
		julian += ( int(current.in_days())/1000 - 1)*1000;
	}
	
	MJD startdate = MJD(julian,int(framesec),(framesec-int(framesec)));
  
	return startdate;
}

MJD dsp::Mark5File::Mark5_stream_frame_time32(double offset)
{
  char nibs[12];
  double framemjd, framesec;
  MJD current;

	lseek(fd,4*fileoffset,SEEK_SET);
	int n=read(fd,framebuf32,4*FRAMESIZE);
	if (!n) throw Error (FailedSys,"dsp::Mark5_stream_frame_time32",
		"Error reading next frame");
		
	extractnibbles32(framebuf32+32,0,12,nibs);
	
	framemjd = nibs[0]*100 + nibs[1]*10 + nibs[2];
	framesec = nibs[3]*10000 + nibs[4]*1000 + nibs[5]*100 + nibs[6]*10
                + nibs[7] + nibs[8]/10.0 + nibs[9]/100.0 + nibs[10]/1000.0
		+ nibs[11]/10000.0 ;
		
	framesec -= offset;

	
	current.Construct(time(NULL));
	
	int julian = int(framemjd);  	
	if (int(current.in_days())%1000 >= julian) {
		// 2 most significant digits of 5 digit julian are correct
		julian += ( int(current.in_days())/1000)*1000;
	} else {
		julian += ( int(current.in_days())/1000 - 1)*1000;
	}
	
	MJD startdate = MJD(julian,int(framesec),(framesec-int(framesec)));
  	
	 

	return startdate;
}

void dsp::Mark5File::set_framenum(int myframeno)
{
	framenum = myframeno;
}

int dsp::Mark5File::get_framenum()
{
	return framenum;
}
		
void dsp::Mark5File::set_payloadoffset(int mypayloadoffset)
{
	payloadoffset = mypayloadoffset;
}

int dsp::Mark5File::get_payloadoffset()
{
	return payloadoffset;
}

unsigned long long* dsp::Mark5File::get_modbits64()
{
	return modbits64;
}

unsigned int* dsp::Mark5File::get_modbits32()
{
	return modbits32;
}

int dsp::Mark5File::get_payloadsize()
{
	return PAYLOADSIZE;
}

int* dsp::Mark5File::get_basebits()
{

	return basebit;
}

int dsp::Mark5File::get_fanout()
{
	return fanout;
}


int dsp::Mark5File::get_framesize()
{	
	return FRAMESIZE;
}


int dsp::Mark5File::get_first_frameNo()
{
	return fileoffset;
}

int dsp::Mark5File::get_bitsperword()
{
	return bitsperword;
}


#if 0

int64 dsp::Mark5File::fstat_file_ndat (uint64 tailer_bytes)
{	struct stat file_stats;

	if (fstat(fd, &file_stats) != 0)
		throw Error (FailedSys,"dsp::Mark5File::fstat_file_ndat",
			"Failed fstat call: fstat(%d)",fd);
	
	int64 actual_file_sz = file_stats.st_size - header_bytes;
	uint64 nblocks = actual_file_sz / block_bytes;	
	uint64 extra = actual_file_sz % block_bytes;
	
	uint64 block_data_bytes = PAYLOADSIZE * sizeof(uint64);	
		// no of bytes of actual data per block
	uint64 data_bytes = nblocks * block_data_bytes;
	
	if (extra > block_header_bytes) {
		extra -= block_header_bytes;
	     	if (extra > block_data_bytes) extra = block_data_bytes;
		data_bytes += extra;
	}

	uint64 bits_per_samp =
	info.get_nchan()*info.get_npol()*info.get_nbit();
	
	if (verbose) {
		cerr << "Mark5File::fstat_file_ndat: file contains ";
		cerr << (data_bytes*8)/bits_per_samp << " samples" << endl;
	}
	
	return (data_bytes*8)/bits_per_samp;
}

int64 dsp::Mark5File::load_bytes (unsigned char* buffer, uint64 bytes)
{	
	if (verbose) cerr << "Mark5File::load_bytes() nbytes =" << bytes << endl;
	
	uint64 block_data_bytes=PAYLOADSIZE*sizeof(uint64);
	uint64 to_load = bytes;

	// find curr_block_byte by doing a findfirstframe (ie from 1st principles)
	if (read( fd, framebuf,FRAMESIZE*sizeof(uint64)) != FRAMESIZE+1024)
		throw Error (FailedSys,"Mark5File::load_bytes",
			"cannot read %d bytes to find frame boundary",FRAMESIZE+1024);

	lseek(fd,-FRAMESIZE*sizeof(uint64),SEEK_CUR);	
			// rewind the file by the amount just read
			
	int posn = findfirstframe(FRAMESIZE);
	if (posn < 0)
		throw Error(InvalidState,"Mark5File::load_bytes",
			"Frame boundary not found");
			
	// no of 64-bit words til start of new frame
	uint64 words_remaining = (posn/2 + 96) - (PAYLOADSIZE-FRAMESIZE);
	if (words_remaining > 0 ) {
		curr_block_byte =  (FRAMESIZE-words_remaining)*sizeof(uint64);
	} else {
		// the data stream starts inside a header so move the file
		// pointer beyond it
		lseek(fd,-words_remaining*sizeof(uint64),SEEK_SET);
		curr_block_byte=0;
		
		// if words_remaining = 0 then we're on the frame boundary
	}
	
	
	while (to_load) {
		uint64 to_read = block_data_bytes - curr_block_byte;
	
		if (to_read > to_load) to_read = to_load;
		
		ssize_t bytes_read = read(fd,buffer,to_read);
		
		if (bytes_read < 0) 
			throw Error (FailedSys,"Mark5File::load_bytes","read(%d)",fd);
			
		to_load -= bytes_read;
		buffer  += bytes_read;
		curr_block_byte += bytes_read; 	
	
		if (curr_block_byte == block_data_bytes) {
		  if (lseek(fd,block_header_bytes,SEEK_CUR) < 0)
			throw Error (FailedSys,"Mark5File::load_bytes","seek(%d)",fd);
			
		  curr_block_byte=0;
		}
		
		// probably the end of the file
		if (uint64(bytes_read) < to_read)
			break;
	}
	
	
	return bytes - to_load;
}

int64 dsp::Mark5File::seek_bytes (uint64 nbytes)
{
	if (verbose)
		cerr << "Mark5File::seek_bytes nbytes=" << nbytes << endl;
		
	if (fd<0)
		throw Error (InvalidState,"Mark5File::seek_bytes","invalid fd");
		
	uint64 block_data_bytes = PAYLOADSIZE * sizeof(uint64);
	uint64 current_block = nbytes/block_data_bytes;
	curr_block_byte = nbytes % block_data_bytes;
	
	uint64 tot_header_bytes = current_block * block_header_bytes;
		// nb mark5 files have no tailer bytes
	uint64 to_byte = nbytes + header_bytes + tot_header_bytes;
	
	if (lseek(fd,to_byte,SEEK_SET) < 0)
		throw Error (FailedSys,"Mark5File::seek_bytes",
			"lseek("UI64")", to_byte);
	if (verbose) {
		cerr << "Mark5File::seek_set to nbytes " << nbytes;
		cerr << " has moved file pointer to position " << to_byte << endl;
	}
	return nbytes;
}

#endif

