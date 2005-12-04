#include <typeinfo>

#include "Error.h"

#include "dsp/Mark5Unpacker.h"
#include "dsp/Mark5File.h"

// modelled on CPSR2_8bitunpacker.C

//! Constructor
//dsp::Mark5Unpacker::Mark5Unpacker (const char* name) : HistUnpacker (name)
dsp::Mark5Unpacker::Mark5Unpacker (const char* _name) : HistUnpacker (_name)
{

}

bool dsp::Mark5Unpacker::matches (const Observation* observation)
{

	return observation->get_machine() == "Mark5" 
	   && observation->get_nbit()==2
	   && observation->get_state() == Signal::Nyquist
	   && observation->get_npol() == 2;

	// NB VLBA data is Nyquist sampled.
}

void dsp::Mark5Unpacker::unpack()
{
	// Bit Stream in input?

	
	const uint64 ndat = input->get_ndat();
	const int nchan=input->get_nchan();      
	const int npol=input->get_npol();

	const float lut[4]={-3.3359,1.0,-1.0,3.3359};
	int o;
	uint64 i, j;
	int f;
	int m, s;
	unsigned long long p;
	

	 if (verbose) {
		cerr << "In Mark5Unpacker with ndat " << ndat << " samples" << endl;
		cerr << npol << " polarisations and " << nchan << " spectral channels" << endl;
	 }
	
	// /////////////////////////////////////////////////
	//
	//   Remaining to do:
	//		 ---
	//
	// /////////////////////////////////////////////////
	
	Mark5File* file = get_Input<Mark5File>();

	int bitsperword = file->get_bitsperword();

	// cerr << "File is a " << bitsperword << "-bit file " << endl;

	int fanout = file->get_fanout();
	int *basebit = file->get_basebits();
	uint64 PAYLOADSIZE = (uint64)(file->get_payloadsize());
	
	bool on_frame_boundary = false;
	bool done = false;
	

	uint64 offset=0;

	int chan, ipol;
	int sbase[npol*nchan];
	int mbase[npol*nchan];

	
	float* data[npol * nchan];
			// get_datptr(chan,pol) chan goes from 0 to 3 and pol from 0 to 1.

	for (ipol = 0 ; ipol < npol ; ipol++) {
	 for (chan=0; chan<nchan; chan++) {
		sbase[ipol+2*chan]=basebit[ipol+2*chan];
		mbase[ipol+2*chan]=basebit[ipol+2*chan]+2*fanout; 
		data[ipol + 2*chan] = output->get_datptr(chan,ipol);
	 } // for chan 
	}  // for ipol


	// 1. find out where we are relative to the start of a frame.
	//    since the start time is set to the beginning of the first, 
	//    full frame, we should start reading from that.

	uint64 words_read = input->get_input_sample()/fanout; // no. of words read so far
	uint64 words_read_this_frame = words_read % PAYLOADSIZE;
	
	if (verbose) cerr << "words_read =" << words_read << ", words_read_this_frame = " << words_read_this_frame;
	uint64 words_to_be_read = PAYLOADSIZE-words_read_this_frame; 
			// no. of words remaining to be read this frame
	if (verbose) cerr << " words to be read =" << words_to_be_read << endl;
	on_frame_boundary = (words_to_be_read == PAYLOADSIZE);

	// 2. perform XOR with modbits on the data and read it in until we hit a frame boundary
	//    or the end of the datastream (ie less than ndat)

if (bitsperword==64) {
	unsigned long long *modbits64 = file->get_modbits64();
	unsigned long long *payload=0;
		// reinterpret the data buffer as a pointer to a bunch of unsigned long longs
	const unsigned long long *buf = reinterpret_cast<const unsigned long long*>(input->get_rawptr());  
	unsigned long long* hold;
	payload = (unsigned long long*)buf;
			
  // framebuf points to the beginning of the frame and payload to the beginning 
  // of the data in that frame
	
	
	while (!done) {	
			
	
			
		if (on_frame_boundary) {
			// we're at the start of a frame:
			// process it  until we
				// 1. run to the end of the frame
				// 2. run out of samples
				
		//	cerr << "framebuf = " << framebuf << " payload = " << payload << endl;
			
			if ((hold = new unsigned long long[PAYLOADSIZE]) == NULL)
				throw Error (FailedSys,"Mark5Unpacker::unpacker",
					"cannot allocate memory for temporary array");
					
			if (verbose) cerr << "On frame boundary" << endl;
			// make a copy of a bit of the buffer so we don't overwrite the BitStream	
			
			for (j = 0; j < PAYLOADSIZE; j++) {
				if ( offset+j*fanout >= ndat) {
					if (verbose) cerr << "Break at j = " << j; 
					break;  
					// we've reached the end of the data stream
				}
				hold[j] = payload[j];
				hold[j] ^= modbits64[j];
				// cerr << "j = " << j << " ndat = " << ndat << endl;
			}
					
		 	for (i = 0 ; i < PAYLOADSIZE*fanout ; i += fanout) {
		
			     o = i/fanout;

			     p=hold[o];	

				for (f=0;f< fanout; f++) {

					  for (chan = 0 ; chan < nchan ; chan++) {
	 				  for (ipol = 0 ; ipol<2; ipol++) {	
					  	// assume that every 2nd channel corresponds to the 
						// other poln at the same frequency.
						s=sbase[ipol+2*chan]+2*f;
						m=mbase[ipol+2*chan]+2*f;
						if (i+f+offset >= ndat) {
							done=true;
							break;
						} else {
						   data[ipol+2*chan][i+f+offset]=lut[((p>>s) & 1) + (((p>>m)&1) <<1)];
						}			
					} // for ipol
					} // for chan		
				} // for f

			
				// o++; (now incremented by i/fanout)
			} // for i;	
				
			delete[] hold;
			
			offset += PAYLOADSIZE*fanout;
			/*		
			if (verbose) {
				cerr << "Offset = " << offset << " Ndat   = " << ndat << endl;
				if (done) cerr << "Done is true " << endl;
				}
			*/
			
			if (!done) payload += PAYLOADSIZE;
				//  advance payload by PAYLOADSIZE records.
					

			
		} else {
			// we're not at the start of a frame.
			// This code should execute at most once per unpack

			hold = new unsigned long long[PAYLOADSIZE];

			if (verbose) {
				cerr << "Mark5Unpacker::unpack - In !on_frame_boundary code ";
				cerr << "with " << words_to_be_read << " remaining words to be read" << endl;
			}
			
			for (j=0;j<words_to_be_read; j++) {
				if ( (uint64)(j*fanout) >= ndat) break;  // we've reached the end of the data stream
				hold[j]  = payload[j];
				hold[j] ^= modbits64[j+words_read_this_frame];
				
			}
				
		 	for (i = 0 ; i < words_to_be_read*fanout ; i += fanout) {
		
				o = i/fanout;
				p=hold[o];
					
				for (f=0;f< fanout; f++) {

					  for (chan = 0 ; chan < nchan ; chan++) {
	 				  for (ipol = 0 ; ipol<2; ipol++) {	
					  	// assume that every 2nd channel corresponds to the 
						// other poln at the same frequency.
						s=sbase[ipol+2*chan]+2*f;
						m=mbase[ipol+2*chan]+2*f;
						if ((uint64)(i+f) >= ndat) {
							done = true;
							break;
						} else {
					  data[ipol+2*chan][i+f]=lut[((p>>s) & 1) + (((p>>m)&1) <<1)];
						}
					   } // for ipol
					  } // for chan		
				} // for f
								
				// o++; (now incremented by i/fanout)
			} // for i;				
				
			offset = words_to_be_read*fanout;
			delete[] hold;
			
			if (!done) {
				payload  = (unsigned long long*)buf + words_to_be_read;  
						// we should now be on a frame boundary
				on_frame_boundary = true;
			}
		}
			
		
	 }  // while (!done)
}  else 
if (bitsperword==32) {
	   	unsigned int *modbits32 = file->get_modbits32();
		unsigned int *payload=0;
		const unsigned int *buf = reinterpret_cast<const unsigned int*>(input->get_rawptr());
		unsigned int* hold;
		payload = (unsigned int*)buf;

	while (!done) {	
			
	
			
		if (on_frame_boundary) {
			// we're at the start of a frame:
			// process it  until we
				// 1. run to the end of the frame
				// 2. run out of samples
				
		//	cerr << "framebuf = " << framebuf << " payload = " << payload << endl;
			
			if ((hold = new unsigned int[PAYLOADSIZE]) == NULL)
				throw Error (FailedSys,"Mark5Unpacker::unpacker",
					"cannot allocate memory for temporary array");
					
			if (verbose) cerr << "On frame boundary" << endl;
			// make a copy of a bit of the buffer so we don't overwrite the BitStream	
			
			for (j = 0; j < PAYLOADSIZE; j++) {
				if ( offset+j*fanout >= ndat) {
					if (verbose) cerr << "Break at j = " << j; 
					break;  
					// we've reached the end of the data stream
				}
				hold[j] = payload[j];
				hold[j] ^= modbits32[j];
				// cerr << "j = " << j << " ndat = " << ndat << endl;
			}
					
		 	for (i = 0 ; i < PAYLOADSIZE*fanout ; i += fanout) {
		
			     o = i/fanout;

			     p=hold[o];	

				for (f=0;f< fanout; f++) {

					  for (chan = 0 ; chan < nchan ; chan++) {
	 				  for (ipol = 0 ; ipol<2; ipol++) {	
					  	// assume that every 2nd channel corresponds to the 
						// other poln at the same frequency.
						s=sbase[ipol+2*chan]+2*f;
						m=mbase[ipol+2*chan]+2*f;
						if (i+f+offset >= ndat) {
							done=true;
							break;
						} else {
						   data[ipol+2*chan][i+f+offset]=lut[((p>>s) & 1) + (((p>>m)&1) <<1)];
						}			
					} // for ipol
					} // for chan		
				} // for f
			
				// o++; (now incremented by i/fanout)
			} // for i;	
				
			delete[] hold;
			
			offset += PAYLOADSIZE*fanout;
			/*		
			if (verbose) {
				cerr << "Offset = " << offset << " Ndat   = " << ndat << endl;
				if (done) cerr << "Done is true " << endl;
				}
			*/
			
			if (!done) payload += PAYLOADSIZE;
				//  advance payload by PAYLOADSIZE records.
					

			
		} else {
			// we're not at the start of a frame.
			// This code should execute at most once per unpack

			hold = new unsigned int[PAYLOADSIZE];

			if (verbose) {
				cerr << "Mark5Unpacker::unpack - In !on_frame_boundary code ";
				cerr << "with " << words_to_be_read << " remaining words to be read" << endl;
			}
			
			for (j=0;j<words_to_be_read; j++) {
				if ( (uint64)(j*fanout) >= ndat) break;  // we've reached the end of the data stream
				hold[j]  = payload[j];
				hold[j] ^= modbits32[j+words_read_this_frame];
				
			}
				
		 	for (i = 0 ; i < words_to_be_read*fanout ; i += fanout) {
		
				o = i/fanout;
				p=hold[o];
					
				for (f=0;f< fanout; f++) {

					  for (chan = 0 ; chan < nchan ; chan++) {
	 				  for (ipol = 0 ; ipol<2; ipol++) {	
					  	// assume that every 2nd channel corresponds to the 
						// other poln at the same frequency.
						s=sbase[ipol+2*chan]+2*f;
						m=mbase[ipol+2*chan]+2*f;
						if ((uint64)(i+f) >= ndat) {
							done = true;
							break;
						} else {
					  data[ipol+2*chan][i+f]=lut[((p>>s) & 1) + (((p>>m)&1) <<1)];
						}
					   } // for ipol
					  } // for chan		
				} // for f
								
				// o++; (now incremented by i/fanout)
			} // for i;				
				
			offset = words_to_be_read*fanout;
			delete[] hold;
			
			if (!done) {
				payload  = (unsigned int*)buf + words_to_be_read;  
						// we should now be on a frame boundary
				on_frame_boundary = true;
			}
		}
			
		
	 }  // while (!done)



}  // is 64/32-bit



#if 0
	
	// some code for test purposes 
	FILE* ftest;
	if ((ftest=fopen("test","a")) == NULL) 
		throw Error (FailedSys,"Mark5Unpacker::unpack","Cannot open output file");
	for (uint64 c=0;c<ndat;c++) {
		fprintf(ftest,"%f ",data[0][c]);
	}
	fclose(ftest);
	
	if (ndat+ input->get_input_sample() > 400000) exit(1);
	// throw Error (FailedSys,"Mark5Unpacker::unpack","deliberate exit after first read");

#endif

}

int dsp::Mark5Unpacker::find_next_frame64(int samples,const unsigned long long* buf)
{
	int i,j;
	unsigned int *dat;
	
	// This routine returns a position that is 64 64-bit words into the header!
	// subtract 128 from its output to get the position of the start of the header.
	
	dat=(unsigned int *)buf;
	
	for (i=2;i<2*samples-64;i++) {
		if (dat[i-1] || dat[i-2]) continue;
		for (j=0;j<64;j++) if (dat[i+j] != 0xFFFFFFFF) break;
		if (j==64) return i;
	}
	return -1;
}
