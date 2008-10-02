/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SigProcDigitizer.h"

//! Default constructor
dsp::SigProcDigitizer::SigProcDigitizer () : Digitizer ("SigProcDigitizer")
{
	nbit = 8;
}

//! Set the number of bits per sample
void dsp::SigProcDigitizer::set_nbit (unsigned _nbit)
{
	switch(_nbit){

		case 1:
		case 2:
		case 4:
		case 8:
			nbit=_nbit;
			break;
		default:
			throw Error (InvalidState, "dsp::SigProcDigitizer::set_nbit",
					"only 8 bit sampling implemented");
			break;
	}
}

unsigned dsp::SigProcDigitizer::get_nbit () const
{
	return nbit;
}

/*! 
  This method must tranpose the data from frequency major order to
  time major order.  It is assumed that ndat > 4 * nchan, and therefore
  stride in output time is smaller than stride in input frequency.

  If this condition isn't true, then the nesting of the loops should
  be inverted.
 */
void dsp::SigProcDigitizer::pack ()
{
	if (input->get_npol() != 1)
		throw Error (InvalidState, "dsp::SigProcDigitizer::pack",
				"cannot handle npol=%d", input->get_npol());

	// the number of frequency channels
	const unsigned nchan = input->get_nchan();

	// the number of time samples
	const uint64 ndat = input->get_ndat();

	unsigned char* outptr = output->get_rawptr();

	float digi_mean=0;
	float digi_sigma=6;
	float digi_scale=0;
	int digi_max=0;
	int digi_min=0;
	int bit_counter=0;
	int samp_per_byte = 8/nbit;

	switch (nbit){
		case 1:
			digi_mean=0.5;
			digi_scale=1;
			digi_min = 0;
			digi_max = 1;
			break;
		case 2:
			digi_mean=1.5;
			digi_scale=1;
			digi_min = 0;
			digi_max = 3;
			break;
		case 4:
			digi_mean=7.5;
			digi_scale= digi_mean / digi_sigma;
			digi_min = 0;
			digi_max = 15;
			break;
		case 8:
			digi_mean=127.5;
			digi_scale= digi_mean / digi_sigma;
			digi_min = 0;
			digi_max = 255;
			break;
	}



	bool flip_band = input->get_bandwidth() > 0;
	if (flip_band)
		output->set_bandwidth(-input->get_bandwidth());

	output->set_nbit(nbit);


	/*
	   TFP mode
	 */

	switch (input->get_order()){

		case TimeSeries::OrderTFP:
			{
				const float* inptr = input->get_dattfp();

				for(uint64 idat=0; idat < ndat; idat++){

					for(unsigned ichan=0; ichan < nchan; ichan++){
						unsigned inChan = ichan;
						if (flip_band)
							inChan = (nchan-ichan-1);

						int result = int( (inptr[idat*nchan + inChan] * digi_scale) + digi_mean +0.5 );

						// clip the result at the limits
						if (result < digi_min)
							result = digi_min;

						if (result > digi_max)
							result = digi_max;

						switch (nbit){
							case 1:
							case 2:
							case 4:
								bit_counter = ichan % (samp_per_byte);

								if(bit_counter==0)outptr[idat*(int)(nchan/samp_per_byte)
									+ (int)(ichan/samp_per_byte)]=(unsigned char)0;
								outptr[idat*(int)(nchan/samp_per_byte)
									+ (int)(ichan/samp_per_byte)] += ((unsigned char) (result)) << (bit_counter*nbit);
								break;
							case 8:
								outptr[idat*nchan + ichan] = (unsigned char) result;
								break;
						}


					}
				}

				return;
			}
		case TimeSeries::OrderPFT:
			{
				for (unsigned ichan=0; ichan < nchan; ichan++)
				{
					const float* inptr;
					if (flip_band)
						inptr = input->get_datptr (nchan-ichan-1);
					else
						inptr = input->get_datptr (ichan);

					for (uint64 idat=0; idat < ndat; idat++)
					{
						int result = int( (inptr[idat] * digi_scale) + digi_mean +0.5 );

						// clip the result at the limits
						if (result < digi_min)
							result = digi_min;

						if (result > digi_max)
							result = digi_max;



						switch (nbit){
							case 1:
							case 2:
							case 4:
								bit_counter = ichan % (samp_per_byte);

								if(bit_counter==0)outptr[idat*(int)(nchan/samp_per_byte) 
									+ (int)(ichan/samp_per_byte)]=(unsigned char)0;
								outptr[idat*(int)(nchan/samp_per_byte) 
									+ (int)(ichan/samp_per_byte)] += ((unsigned char) (result)) << (bit_counter*nbit);
								break;
							case 8:
								outptr[idat*nchan + ichan] = (unsigned char) result;
								break;
						}
					}
				}
				return;
			}
		default:
			throw Error (InvalidState, "dsp::SigProcDigitizer::operate",
					"Can only operate on data ordered FTP or PFT.");

	}
}


