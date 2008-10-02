/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Rescale.h"

using namespace std;

dsp::Rescale::Rescale ()
	: Transformation<TimeSeries,TimeSeries> ("Rescale", anyplace)
{
	iblock = 0;
	sample = 0;
	bandpass_monitor = new dsp::BandpassMonitor(); 
	prev_mean = NULL;
	prev_var = NULL;
}


/*!
  \pre input TimeSeries must contain detected data
  */
void dsp::Rescale::transformation ()
{
	if (verbose)
		cerr << "dsp::Rescale::transformation" << endl;

	const uint64   input_ndat  = input->get_ndat();
	const unsigned input_ndim  = input->get_ndim();
	const unsigned input_npol  = input->get_npol();
	const unsigned input_nchan = input->get_nchan();
	
	//fprintf(stderr,"input_ndat %d \n",input_ndat);
	//fprintf(stderr,"input_nchan %d \n",input_nchan);

	if (input_ndim != 1)
		throw Error (InvalidState, "dsp::Rescale::transformation",
				"invalid ndim=%d", input_ndim);

	uint64 output_ndat = input_ndat;

	// prepare the output TimeSeries
	output->copy_configuration (input);

	if (output != input)
		output->resize (output_ndat);
	else
		output->set_ndat (output_ndat);

	if (!output_ndat)
		return;

	float* freqs = new float[input_nchan];
	for (unsigned ichan=0; ichan < input_nchan; ichan++)
		freqs[ichan] = input->get_centre_frequency(ichan);

	for (unsigned ipol=0; ipol < input_npol; ipol++) 
	{
		float* mean_bandpass= new float[input_nchan];
		float* variance_bandpass = new float[input_nchan];
		float* rms_bandpass = new float[input_nchan];
		if(prev_mean == NULL){
			prev_mean = new float*[input_npol];
			prev_var = new float*[input_npol];
			for(unsigned p = 0; p < input_npol; p++){
				prev_mean[p] = new float[input_nchan];
				prev_var[p] = new float[input_nchan];
				for (unsigned ichan=0; ichan < input_nchan; ichan++){
					 prev_mean[p][ichan] = -1;
				}
			}
		}
	
		// array to store zero DM time series
		float* zerotime = new float[input_ndat];

		for (unsigned ichan=0; ichan < input_nchan; ichan++)
		{
			const float* in_data = input->get_datptr (ichan, ipol);

			double sum = 0.0;
			double sumsq = 0.0;

			for (uint64 idat=0; idat < input_ndat; idat++)
			{
				sum += in_data[idat];
				sumsq += in_data[idat] * in_data[idat];
				zerotime[idat] += in_data[idat];
			}

			double mean = sum / input_ndat;
			double variance = sumsq/input_ndat - mean*mean;
			double rms = sqrt(variance);
			
			
                        mean_bandpass[ichan] = (float)mean;
                        variance_bandpass[ichan] = (float)variance;
                        rms_bandpass[ichan] = (float)rms;

//			if( prev_mean[ipol][ichan] > 0){
//				if(mean > prev_mean[ipol][ichan]*1.005)mean = prev_mean[ipol][ichan]*1.005;
//				if(mean < prev_mean[ipol][ichan]*0.995)mean = prev_mean[ipol][ichan]*0.995;
//				if(variance > prev_var[ipol][ichan]*1.005)variance = prev_var[ipol][ichan]*1.005;
//				if(variance < prev_var[ipol][ichan]*0.995)variance = prev_var[ipol][ichan]*0.995;
//				
//				mean = (mean + prev_mean[ipol][ichan])/2.0;
//				variance = (variance + prev_var[ipol][ichan])/2.0;
//			}
//
//			prev_mean[ipol][ichan] = mean;
//			prev_var[ipol][ichan] = variance;



//			if( prev_mean[ipol][ichan] < 0){
//				 prev_mean[ipol][ichan] = mean;
//				 prev_var[ipol][ichan] = variance;
//			} else {
//				mean = prev_mean[ipol][ichan];
//				variance =  prev_var[ipol][ichan];
//			}


			float scale = 1.0/sqrt(variance);


			float* out_data = output->get_datptr (ichan, ipol);

			for (uint64 idat=0; idat < output_ndat; idat++)
				out_data[idat] = (in_data[idat] - mean) * scale;
		}
		//bandpass_monitor->append(sample,sample+input_ndat,ipol,input_nchan,mean_bandpass,variance_bandpass, freqs);
		//also include zero dm time series
		bandpass_monitor->append(sample,sample+input_ndat,ipol,input_nchan,mean_bandpass,variance_bandpass,rms_bandpass,freqs,zerotime);

	}
	sample += input_ndat;
	iblock++;
}

