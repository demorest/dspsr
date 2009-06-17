/***************************************************************************
 *
 *   Copyright (C) 2006 by Eric Plum
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/FadcUnpacker.h"
#include "Error.h"

#include <fstream>
#include "fadc_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

using namespace std;

struct two_bit_swin{
  unsigned lRe0  :2;
  unsigned lIm0  :2;
  unsigned rRe0  :2;
  unsigned rIm0  :2;
  unsigned lRe1  :2;
  unsigned lIm1  :2;
  unsigned rRe1  :2;
  unsigned rIm1  :2;
  unsigned lRe2  :2;
  unsigned lIm2  :2;
  unsigned rRe2  :2;
  unsigned rIm2  :2;
  unsigned lRe3  :2;
  unsigned lIm3  :2;
  unsigned rRe3  :2;
  unsigned rIm3  :2;
};

struct four_bit_swin{
  unsigned lRe0  :4;
  unsigned lIm0  :4;
  unsigned rRe0  :4;
  unsigned rIm0  :4;
  unsigned lRe1  :4;
  unsigned lIm1  :4;
  unsigned rRe1  :4;
  unsigned rIm1  :4;
};

struct eight_bit_swin{
  unsigned lRe0  :8;
  unsigned lIm0  :8;
  unsigned rRe0  :8;
  unsigned rIm0  :8;
};

//! Constructor
dsp::FadcUnpacker::FadcUnpacker (const char* name) : HistUnpacker (name)
{
}

//! Specialize the unpacker to the Observation                                                                                        
void dsp::FadcUnpacker::match (const Observation* observation)                                                             
{                                                                                                                                     
  // number of bins in the histgram (I guess) = 2^nbit
  if (observation->get_nbit() == 2) set_nsample (4);
  if (observation->get_nbit() == 4) set_nsample (16);
  if (observation->get_nbit() == 8) set_nsample (256);
}                                   
        
bool dsp::FadcUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Fadc" 
    && (/*observation->get_nbit() == 2  || */ observation->get_nbit() == 4  ||  observation->get_nbit() == 8)
    && observation->get_state() == Signal::Analytic;
}

void dsp::FadcUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const unsigned int npol = input->get_npol();

cerr<<"ndat = "<<ndat<<'\n';
cerr<<"npol = "<<npol<<'\n';

  if (npol==2)
  {
    float* into_LCP = output->get_datptr (0, 0);
    float* into_RCP = output->get_datptr (0, 1);
    unsigned long* hist_LCP = get_histogram (0);   //! uint64_t would be better
    unsigned long* hist_RCP = get_histogram (1);
  
    uint64_t sample=0;

       const char* from_char = reinterpret_cast<const char*>(input->get_rawptr());
 
    
  // 2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   
    if (input->get_nbit() == 2)
    {
       struct two_bit_swin* from = (struct two_bit_swin*) from_char;

       // from: array of 4byte sized structures containing 4x 4 unsigned 2bit values (LCP re, LCP im, RCP re, RCP im)
  
       // the data bytes containing 4 2-bit numbers
       // 1: LCP real, 2:LCP imag, 3: RCP real, 4:RCP imag
       for (uint64_t fourbyte = 0; fourbyte < ndat/4; fourbyte++)
       {
            // fill histogram, original values are unsigned, i.e. 0, 1, 2, 3
            // read 2bit values, do offset, convert to float and write out
            // e.g. LCP output: re, im, re, im, ...
            hist_LCP[ from[fourbyte].lRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm0 ]++;    // LCP im       
            hist_RCP[ from[fourbyte].rRe0 ]++;  // RCP re
            hist_RCP[ from[fourbyte].rIm0 ]++;   // RCP im
            into_LCP[2*sample]  = (float)   from[fourbyte].lRe0  -1.5;  // LCP re
            into_LCP[2*sample+1]= (float) from[fourbyte].lIm0  -1.5;  // LCP im       
            into_RCP[2*sample]  = (float)   from[fourbyte].rRe0  -1.5;  // RCP re
            into_RCP[2*sample+1]= (float) from[fourbyte].rIm0  -1.5;  // RCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe1 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm1 ]++;    // LCP im       
            hist_RCP[ from[fourbyte].rRe1 ]++;  // RCP re
            hist_RCP[ from[fourbyte].rIm1 ]++;   // RCP im
            into_LCP[2*sample]  = (float)   from[fourbyte].lRe1  -1.5;  // LCP re
            into_LCP[2*sample+1]= (float) from[fourbyte].lIm1  -1.5;  // LCP im       
            into_RCP[2*sample]  = (float)   from[fourbyte].rRe1  -1.5;  // RCP re
            into_RCP[2*sample+1]= (float) from[fourbyte].rIm1  -1.5;  // RCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe2 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm2 ]++;    // LCP im       
            hist_RCP[ from[fourbyte].rRe2 ]++;  // RCP re
            hist_RCP[ from[fourbyte].rIm2 ]++;   // RCP im
            into_LCP[2*sample]  = (float)   from[fourbyte].lRe2  -1.5;  // LCP re
            into_LCP[2*sample+1]= (float) from[fourbyte].lIm2  -1.5;  // LCP im       
            into_RCP[2*sample]  = (float)   from[fourbyte].rRe2  -1.5;  // RCP re
            into_RCP[2*sample+1]= (float) from[fourbyte].rIm2  -1.5;  // RCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe3 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm3 ]++;    // LCP im       
            hist_RCP[ from[fourbyte].rRe3 ]++;  // RCP re
            hist_RCP[ from[fourbyte].rIm3 ]++;   // RCP im
            into_LCP[2*sample]  = (float)   from[fourbyte].lRe3  -1.5;  // LCP re
            into_LCP[2*sample+1]= (float) from[fourbyte].lIm3  -1.5;  // LCP im       
            into_RCP[2*sample]  = (float)   from[fourbyte].rRe3  -1.5;  // RCP re
            into_RCP[2*sample+1]= (float) from[fourbyte].rIm3  -1.5;  // RCP im
            sample++; 

          }  // end of loop over samples
      }  // end of 2bit unpacking
    
    
    // 4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   
    else if (input->get_nbit() == 4)
    {
       struct four_bit_swin* from = (struct four_bit_swin*) from_char;

       // from: array of 4byte sized structures containing 2x 4 unsigned 4bit values (LCP re, LCP im, RCP re, RCP im)

       // the data bytes containing 2 4-bit numbers
       // 1: LCP real, 2:LCP imag; 1: RCP real, 2:RCP imag
       for (uint64_t fourbyte = 0; fourbyte < ndat/2; fourbyte++)
       {
            // fill histogram, original values are unsigned, i.e. 0, 1, 2, 3... 7
            // read 4bit values, do offset, convert to float and write out
            // e.g. LCP output: re, im, re, im, ...
            hist_LCP[ from[fourbyte].lRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm0 ]++;    // LCP im       
            hist_RCP[ from[fourbyte].rRe0 ]++;  // RCP re
            hist_RCP[ from[fourbyte].rIm0 ]++;   // RCP im
            into_LCP[2*sample]  = (float)   from[fourbyte].lRe0  -7.5;  // LCP re
            into_LCP[2*sample+1]= (float) from[fourbyte].lIm0  -7.5;  // LCP im       
            into_RCP[2*sample]  = (float)   from[fourbyte].rRe0  -7.5;  // RCP re
            into_RCP[2*sample+1]= (float) from[fourbyte].rIm0  -7.5;  // RCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe1 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm1 ]++;    // LCP im       
            hist_RCP[ from[fourbyte].rRe1 ]++;  // RCP re
            hist_RCP[ from[fourbyte].rIm1 ]++;   // RCP im
            into_LCP[2*sample]  = (float)   from[fourbyte].lRe1  -7.5;  // LCP re
            into_LCP[2*sample+1]= (float) from[fourbyte].lIm1  -7.5;  // LCP im       
            into_RCP[2*sample]  = (float)   from[fourbyte].rRe1  -7.5;  // RCP re
            into_RCP[2*sample+1]= (float) from[fourbyte].rIm1  -7.5;  // RCP im
            sample++;
          }  // end of loop over samples
      }  // end of 4bit unpacking
    
    
    
    // 8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   
    else if (input->get_nbit() == 8)
    {
       struct eight_bit_swin* from = (struct eight_bit_swin*) from_char;

       // from: array of 4byte sized structures containing 1x 4 unsigned 8bit values (LCP re, LCP im, RCP re, RCP im)
  
       // the data bytes containing 8-bit numbers
       // 1: LCP real, 2:LCP imag, 3: RCP real, 4:RCP imag   - four bytes
       for (uint64_t fourbyte = 0; fourbyte < ndat/1; fourbyte++)
       {
            // fill histogram, original values are unsigned, i.e. 0, 1, 2, 3... 255
            // read 8bit values, do offset, convert to float and write out
            // e.g. LCP output: re, im, re, im, ...
            hist_LCP[ from[fourbyte].lRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm0 ]++;    // LCP im       
            hist_RCP[ from[fourbyte].rRe0 ]++;  // RCP re
            hist_RCP[ from[fourbyte].rIm0 ]++;   // RCP im
            into_LCP[2*sample]  = (float)   from[fourbyte].lRe0  -127.5;  // LCP re
            into_LCP[2*sample+1]= (float) from[fourbyte].lIm0  -127.5;  // LCP im       
            into_RCP[2*sample]  = (float)   from[fourbyte].rRe0  -127.5;  // RCP re
            into_RCP[2*sample+1]= (float) from[fourbyte].rIm0  -127.5;  // RCP im
            sample++;
          }  // end of loop over samples
      }  // end of 8bit unpacking
      else 
      {
        cerr<<"FadcUnpacker: I can only unpack 2/4/8 bit data (2 polarizations). This data has: "<<(input->get_nbit())<<" bit values.\n";
        throw Error (InvalidState, "FadcUnpacker", 
		     "Fatal Error (Cannot unpack data)");
      }
    } // end of 2 polarizations

  else if (npol==1)
  {
    float* into_LCP = output->get_datptr (0, 0);
    unsigned long* hist_LCP = get_histogram (0);   //! uint64_t would be better
  
    uint64_t sample=0;

       const char* from_char = reinterpret_cast<const char*>(input->get_rawptr());
 
    
  // 2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   2bit   
    if (input->get_nbit() == 2)
    {
       struct two_bit_swin* from = (struct two_bit_swin*) from_char;

       // from: array of 4byte sized structures containing 4x 4 unsigned 2bit values (LCP re, LCP im or other polarization)
  
       // the data bytes containing 4 2-bit numbers
       // 1: LCP real, 2:LCP imag, 3: LCP real, 4:LCP imag  or other polarization
       for (uint64_t fourbyte = 0; fourbyte < ndat/8; fourbyte++)
       {
            // read 2bit values, do offset, convert to float and write out
            // e.g. LCP output: re, im, re, im, ...
	    // NOTE: sample counts pairs of samples !!!
            hist_LCP[ from[fourbyte].lRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm0 ]++;    // LCP im       
            hist_LCP[ from[fourbyte].rRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].rIm0 ]++;    // LCP im       
            into_LCP[4*sample+0]= (float)   from[fourbyte].lRe0 -1.5;  // LCP re
            into_LCP[4*sample+1]= (float)   from[fourbyte].lIm0  -1.5;  // LCP im       
            into_LCP[4*sample+2]= (float)   from[fourbyte].rRe0 -1.5;  // LCP re
            into_LCP[4*sample+3]= (float)   from[fourbyte].rIm0  -1.5;  // LCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe1 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm1 ]++;    // LCP im       
            hist_LCP[ from[fourbyte].rRe1 ]++;   // LCP re
            hist_LCP[ from[fourbyte].rIm1 ]++;    // LCP im       
            into_LCP[4*sample+0]= (float)   from[fourbyte].lRe1 -1.5;  // LCP re
            into_LCP[4*sample+1]= (float)   from[fourbyte].lIm1  -1.5;  // LCP im       
            into_LCP[4*sample+2]= (float)   from[fourbyte].rRe1 -1.5;  // LCP re
            into_LCP[4*sample+3]= (float)   from[fourbyte].rIm1  -1.5;  // LCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe2 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm2 ]++;    // LCP im       
            hist_LCP[ from[fourbyte].rRe2 ]++;   // LCP re
            hist_LCP[ from[fourbyte].rIm2 ]++;    // LCP im       
            into_LCP[4*sample+0]= (float)   from[fourbyte].lRe2 -1.5;  // LCP re
            into_LCP[4*sample+1]= (float)   from[fourbyte].lIm2  -1.5;  // LCP im       
            into_LCP[4*sample+2]= (float)   from[fourbyte].rRe2 -1.5;  // LCP re
            into_LCP[4*sample+3]= (float)   from[fourbyte].rIm2  -1.5;  // LCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe3 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm3 ]++;    // LCP im       
            hist_LCP[ from[fourbyte].rRe3 ]++;   // LCP re
            hist_LCP[ from[fourbyte].rIm3 ]++;    // LCP im       
            into_LCP[4*sample+0]= (float)   from[fourbyte].lRe3 -1.5;  // LCP re
            into_LCP[4*sample+1]= (float)   from[fourbyte].lIm3  -1.5;  // LCP im       
            into_LCP[4*sample+2]= (float)   from[fourbyte].rRe3 -1.5;  // LCP re
            into_LCP[4*sample+3]= (float)   from[fourbyte].rIm3  -1.5;  // LCP im
            sample++; 

          }  // end of loop over samples
      }  // end of 2bit unpacking
    
    
    // 4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   4bit   
    else if (input->get_nbit() == 4)
    {
       struct four_bit_swin* from = (struct four_bit_swin*) from_char;

       // from: array of 4byte sized structures containing 2x 4 unsigned 4bit values (LCP re, LCP im, LCP re, LCP im)
  
       // the data bytes containing 2 4-bit numbers
       // 1: LCP real, 2:LCP imag
       for (uint64_t fourbyte = 0; fourbyte < ndat/4; fourbyte++)
       {
            // read 4bit values, do offset, convert to float and write out
            // e.g. LCP output: re, im, re, im, ...
            hist_LCP[ from[fourbyte].lRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm0 ]++;    // LCP im       
            hist_LCP[ from[fourbyte].rRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].rIm0 ]++;    // LCP im       
            into_LCP[4*sample+0]= (float)   from[fourbyte].lRe0  -7.5; // LCP re
            into_LCP[4*sample+1]= (float)   from[fourbyte].lIm0   -7.5;// LCP im       
            into_LCP[4*sample+2]= (float)   from[fourbyte].rRe0  -7.5;  // LCP re
            into_LCP[4*sample+3]= (float)   from[fourbyte].rIm0   -7.5 ;  // LCP im
            sample++;
            hist_LCP[ from[fourbyte].lRe1 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm1 ]++;    // LCP im       
            hist_LCP[ from[fourbyte].rRe1 ]++;   // LCP re
            hist_LCP[ from[fourbyte].rIm1 ]++;    // LCP im       
            into_LCP[4*sample+0]= (float)   from[fourbyte].lRe1  -7.5;  // LCP re
            into_LCP[4*sample+1]= (float)   from[fourbyte].lIm1  -7.5 ;  // LCP im       
            into_LCP[4*sample+2]= (float)   from[fourbyte].rRe1  -7.5;  // LCP re
            into_LCP[4*sample+3]= (float)   from[fourbyte].rIm1  -7.5 ;  // LCP im
            sample++;
          }  // end of loop over samples
      }  // end of 4bit unpacking
    
    
    
    // 8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   8bit   
    else if (input->get_nbit() == 8)
    {
       struct eight_bit_swin* from = (struct eight_bit_swin*) from_char;

       // from: array of 4byte sized structures containing 1x 4 unsigned 8bit values (LCP re, LCP im, LCP re, LCP im)
  
       // the data bytes containing 8-bit numbers
       // 1: LCP real, 2:LCP imag, 3: LCP real, 4:LCP imag   - four bytes
       for (uint64_t fourbyte = 0; fourbyte < ndat/2; fourbyte++)
       {
            hist_LCP[ from[fourbyte].lRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].lIm0 ]++;    // LCP im       
            hist_LCP[ from[fourbyte].rRe0 ]++;   // LCP re
            hist_LCP[ from[fourbyte].rIm0 ]++;    // LCP im       
            into_LCP[4*sample+0]= (float)   from[fourbyte].lRe0  -127.5;  // LCP re
            into_LCP[4*sample+1]= (float)   from[fourbyte].lIm0  -127.5;  // LCP im       
            into_LCP[4*sample+2]= (float)   from[fourbyte].rRe0  -127.5;  // LCP re
            into_LCP[4*sample+3]= (float)   from[fourbyte].rIm0  -127.5;  // LCP im
            sample++;
          }  // end of loop over samples
      }  // end of 8bit unpacking
      else 
      {
        cerr<<"FadcUnpacker: I can only unpack 2/4/8 bit data (1 polarization). This data has: "<<(input->get_nbit())<<" bit values.\n";
        throw Error (InvalidState, "FadcUnpacker", "Fatal Error (Cannot unpack data)");
      }
    } // end of 1 polarization
    else
    {
      cerr<<"FadcUnpacker: I can only unpack data with 1 or 2 polarizations. This data has: "<<npol<<" polarizations.\n";
      throw Error (InvalidState, "FadcUnpacker", "Fatal Error (Cannot unpack data)");
    }
cerr<<"FadcUnpack: finished\n\n";
}  // end of unpack

