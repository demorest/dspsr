#include "Fold.h"
#include "PhaseSeries.h"

#include "polyco.h"
#include "genutil.h"

dsp::Fold::Fold () : Operation ("Fold", outofplace) 
{
  folding_period = 0;
}

//! Set the period at which to fold data (in seconds)
void dsp::Fold::set_folding_period (double _folding_period)
{
  folding_period = _folding_period;
  folding_polyco = 0;
}

//! Get the average folding period
double dsp::Fold::get_folding_period () const
{
  if (folding_polyco)
    return folding_polyco->get_refperiod();
  else
    return folding_period;
}

//! Set the phase polynomial(s) with which to fold data
void dsp::Fold::set_folding_polyco (polyco* _folding_polyco)
{
  folding_polyco = _folding_polyco;
  folding_period = 0.0;
}

void dsp::Fold::operation ()
{
  PhaseSeries* profile = dynamic_cast<PhaseSeries*> (output.get());

  if (!profile)
    throw_str ("Fold::operation output is not a PhaseSeries");

  if (!profile->mixable (*input, nbin))
    throw_str ("Fold::operation cannot mix input with output");

  if (folding_period == 0 && folding_polyco == NULL)
    throw_str ("Fold::operation no folding period or polyco set");
  
  int blocks = input->get_nchan() * input->get_npol();

  sampling_interval = 1.0/input->get_rate();
  start_time = input->get_start_time() + 0.5 * sampling_interval;

  int64 block_size = input->get_datptr (0,1) - input->get_datptr (0,0);
  int64 block_ndat = block_size / input->get_ndim();

  assert (block_size % input->get_ndim() == 0);

  fold (blocks, block_ndat, input->get_ndim(),
	input->get_datptr(), profile->get_datptr(), profile->hits.begin(),
	input->get_ndat());

  profile->integration_length += double(input->get_ndat())*sampling_interval;
}

/*!  This method creates a folding plan and then folds nblock arrays.

   \pre the nbin, sampling_interval, start_time, and folding_period or
   folding_polyco attributes must have been set prior to calling this
   method.

   \param nblock the number of blocks of data to be folded
   \param ndat the number of time samples in each time block
   \param ndim the dimension of each time sample
   \param time base address of nblock contiguous time blocks (of ndat*ndim)
   \param phase base address of nblock contiguous phase blocks (of nbin*ndim)
   \param hits array of nbin phase bin counts
   \param fold_ndat the number of time samples to be folded from each block
*/
void dsp::Fold::fold (unsigned nblock, int64 ndat, unsigned ndim,
		      const float* time, float* phase, unsigned* hits,
		      int64 fold_ndat)
{
  if (fold_ndat == 0)
    fold_ndat = ndat;

  double phi=0, pfold=0;

  if (folding_period != 0) {

    pfold = folding_period;
    phi   = fmod (start_time.in_seconds(), folding_period) / folding_period;
    while (phi<0.0) phi += 1.0;

    if (verbose)
      cerr << "folding::fold CAL period=" << pfold << endl;

  }
  else {

    // find the period and phase at the mid time of the first sample
    pfold = folding_polyco->period(start_time);
    phi = folding_polyco->phase(start_time).fracturns();
    if (phi<0.0) phi += 1.0;
    
    if (verbose)
      cerr << "folding::fold polyco.period=" << pfold << endl;

  }

  double nphi = double (nbin);
  double binspersample = nphi * sampling_interval / pfold;
  phi *= nphi;

  if (verbose)
    cerr << "Fold::fold phase=" << phi << endl;

  //
  // MAKE FOLDING PLAN
  //

  // allocate storage for phase bin plan
  unsigned* binplan = (unsigned*) workingspace (fold_ndat * sizeof(unsigned));

  // counter through time dimension
  int64 idat = 0;
  // counter through phase dimension
  unsigned ibin = 0;
  // counter through space dimension
  unsigned idim = 0;

  for (idat=0; idat < fold_ndat; idat++) {

    ibin = unsigned(phi);
    phi += binspersample;
    if (phi >= nphi) phi -= nphi;

    assert (ibin < nbin);
    binplan[idat] = ibin;

    hits[ibin]++;

  }

  //
  // FOLD ARRAYS
  //

  // pointer through phase dimension
  float *phasep;
  // pointer through dimensions after phase
  float *phdimp;

  unsigned phase_block_size = nbin * ndim;

  // pointer through time dimension
  const float *timep;

  int64 time_block_size = ndat * ndim;

  for (unsigned iblock=0; iblock<nblock; iblock++) {

    timep = time + time_block_size * iblock;
    phasep = phase + phase_block_size * iblock;

    for (idat=0; idat < fold_ndat; idat++) {

      // point to the right phase
      phdimp = phasep + binplan[idat] * ndim;

      // copy the n dimensions
      for (idim=0; idim<ndim; idim++) {
	phdimp[idim] += *timep;
	timep ++;
      }
      
    }
  }
}

