/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Mark5Observation.h"
#include "coord.h"
#include "ascii_header.h"
#include <unistd.h>
#include "fcntl.h"
#include <sys/types.h>

using namespace std;

//modelled on CPSR2_8bitObservation.C

dsp::Mark5_Observation::Mark5_Observation(const char* header)
{
	// For the time being just get all the parameters from a text file
	// The Mk5 files don't contain source information anyway, so
	// we'll always need to read in a separate file

	cerr << "In Mark5_Observation" << endl;

	printf("header = %s\n",header);
	// NB: where do we open the header file or specify its name?
	if (header == NULL)
		throw Error (InvalidState,"Mark5_Observation", "no header string");
	
	//
	// no idea about the size of the data
	//
	
	set_ndat(0);
	
	// ///////////////////////////////////////////////////////////////
	// TELESCOPE
	//
	char hdrstr[1024];
	FILE *ftext;
	
	if ((ftext=fopen(header,"r")) == NULL) 
		throw Error (FailedSys,"dsp::Mark5_Observation",
			"Cannot open header file %s",header);
	
	fread(hdrstr,sizeof(char),1024,ftext);
	
	
	if (ascii_header_get (header,"TELESCOPE","%s",hdrstr) <0)
		throw Error (InvalidState,"Mark5_Observation", "failed read TELESCOPE");
	/* user must specify a telescope whose name is recognised or the telescope
	code */
	
    	set_telescope (hdrstr);

	// ///////////////////////////////////////////////////////////////	
	// SOURCE
	//
	  if (ascii_header_get (header, "SOURCE", "%s", hdrstr) < 0)
	    throw Error (InvalidState,"Mark5_Observation", "failed read SOURCE");

  	set_source (hdrstr);
	// ///////////////////////////////////////////////////////////////
	// FREQ
	//
	// Note that we assign the CENTRE frequency, not the edge of the band
	//  fix this is up in later version
	double freq;
  	if (ascii_header_get (header, "FREQ", "%lf", &freq) < 0)
    	throw Error (InvalidState,"Mark5_Observation", "failed read FREQ");

  	set_centre_frequency (freq);

	// WvS - flag means that even number of channels are result of FFT
	// dc_centred = true;

	// ///////////////////////////////////////////////////////////////
	// BW
	//
	double bw;
	if (ascii_header_get (header, "BW", "%lf", &bw) < 0)
    		throw Error (InvalidState,"Mark5_Observation", "failed read BW");

	set_bandwidth (bw);
	
	
	// ///////////////////////////////////////////////////////////////
	// No. of CHANNELS
	//
	// We're going to read in all four 16 MHz streams at once
	//  --- we'll generalise this later 
	
	set_nchan(4);	
	
	// ///////////////////////////////////////////////////////////////
	// NPOL
	//	
	//  -- generalise this later
	
	set_npol(2);    // read in both polns at once
	
	// ///////////////////////////////////////////////////////////////	
	// NBIT
	//
	
	int scan_nbit;
	if (ascii_header_get (header, "NBIT", "%d", &scan_nbit) < 0)
	   throw Error (InvalidState,"Mark5_Observation", "failed read NBIT");

	set_nbit (scan_nbit);

	// ///////////////////////////////////////////////////////////////	
	// NDIM  --- whether the data are Nyquist or Quadrature sampled
	//
	// VLBA data are Nyquist sampled
	  set_state (Signal::Nyquist);
	  
	
	 //
	 // call this only after setting frequency and telescope
  	 //

	// ///////////////////////////////////////////////////////////////
	//  FANOUT
	//
	if (ascii_header_get (header,"FANOUT","%d",&fanout) < 0)
		throw Error (InvalidState,"Mark5_Observation", "failed read FANOUT");



	// ///////////////////////////////////////////////////////////////
	// TSAMP
	//
	// Note TSAMP is the sampling period in microseconds
	double sampling_interval;
  	if (ascii_header_get (header, "TSAMP", "%lf", &sampling_interval)<0)
    		throw Error (InvalidState,"Mark5_Observation", "failed read TSAMP");

  	/* IMPORTANT: TSAMP is the sampling period in microseconds */
  	sampling_interval *= 1e-6;

  	set_rate (1.0/sampling_interval); 

	// ///////////////////////////////////////////////////////////////
	// FILENAME of the actual MkV data file
	string datafilename;   
	
	 if (ascii_header_get (header,"DATAFILE","%s",hdrstr) < 0 )
	 	throw Error (InvalidState,"Mark5_Observation", "failed read Mark5 data filename");
	
	// ///////////////////////////////////////////////////////////////	  
	// MJD_START
	//
	// look in the actual data file for the mjd of the first frame
	// rather than relying on user input from the header file
	// THIS PARAMETER is SET in Mark5File:: open_file   	 
	
		
	
	// ///////////////////////////////////////////////////////////////
	// CALCULATE the various offsets and sizes
	//
	// PRIMARY  --- what's this???
	
	string prefix="?";    // what prefix should we assign??
	  
	set_mode( tostring(get_nbit()) + "-bit mode" );
	set_machine("Mark5");
	
	// ///////////////////////////////////////////////////////////////
	// RA and DEC
	//
		
	bool has_position = true;
	double ra, dec;

	if (has_position){
    		has_position = (ascii_header_get (header, "RA", "%s", hdrstr) == 1);
    	  //    fprintf(stderr,"1 has_position=%d hdrstr='%s'\n",has_position,hdrstr);
 	 	}

	if (has_position){
		has_position = (str2ra (&ra, hdrstr) == 0);
    	      //fprintf(stderr,"2 has_position=%d ra=%f\n",has_position,ra);
	      }

	if (has_position){
		has_position = (ascii_header_get (header, "DEC", "%s", hdrstr) == 1);
    	      //fprintf(stderr,"3 has_position=%d hdrstr='%s'\n",has_position,hdrstr);
	      }

	if (has_position){
    		has_position = (str2dec2 (&dec, hdrstr) == 0);
    	      //fprintf(stderr,"4 has_position=%d dec=%f\n",has_position,dec);
	      }

	if (!has_position){
		ra = dec = 0.0;
	      //fprintf(stderr,"5 has_position=%d set shit to zero\n",has_position);
  	      }
	      
	coordinates.setRadians(ra,dec);


}


int dsp::Mark5_Observation::get_fanout()
{
	return fanout;
}
