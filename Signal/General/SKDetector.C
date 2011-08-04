/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKDetector.h"

#include "dsp/Unpacker.h"
#include "dsp/Input.h"
#include "dsp/SKLimits.h"

#include <assert.h>
#include <iomanip>

//#define USE_MEGA_THRESHOLDS 1

using namespace std;

dsp::SKDetector::SKDetector () 
  : Transformation<TimeSeries,BitSeries>("SKDetector",outofplace)
{
  M = 0;
  one_sigma = 0;
  n_std_devs = 3;
  upper_thresh = 0;
  lower_thresh = 0;
#ifdef USE_MEGA_THRESHOLDS
  mega_upper_thresh = 0;
  mega_lower_thresh = 0;
#endif
  debugd = 1;
  s_chan = 0;
  e_chan = 0;
  ndat_zapped = 0;
  ndat_zapped_skfb = 0;
#ifdef USE_MEGA_THRESHOLDS
  ndat_zapped_mega = 0;
#endif
  ndat_zapped_fscr = 0;
  ndat_zapped_tscr = 0;
  ndat_total = 0;
  disable_fscr = false;
  disable_tscr = false;
  disable_ft = false;
  input_tscr = 0;
  tscr_M = 0;
  tscr_upper = 0;
  tscr_lower = 0;
}

dsp::SKDetector::~SKDetector ()
{
  if (verbose)
    cerr << "dsp::SKDetector::~SKDetector()" << endl;

  cerr << "Zapped: " 
       << " total=" << (100 * (float) ndat_zapped / (float) ndat_total) << "\%" 
       << " skfb=" << (100 * (float) ndat_zapped_skfb / (float) ndat_total) << "\%"
#ifdef USE_MEGA_THRESHOLDS
       << " mega=" << (100 * (float) ndat_zapped_mega / (float) ndat_total) << "\%"
#endif
       << " tscr=" << (100 * (float) ndat_zapped_tscr / (float) ndat_total) << "\%"
       << " fscr=" << (100 * (float) ndat_zapped_fscr / (float) ndat_total) << "\%"
       << endl;
}

void dsp::SKDetector::set_thresholds (unsigned _M, unsigned _n_std_devs)
{
  M = _M;
  n_std_devs = _n_std_devs;

  if (verbose)
    cerr << "dsp::SKDetector::set_thresholds SKlimits(" << M << ", " << n_std_devs << ")" << endl;
  dsp::SKLimits limits(M, n_std_devs);
  limits.calc_limits();

  upper_thresh = (float) limits.get_upper_threshold();
  lower_thresh = (float) limits.get_lower_threshold();

#ifdef USE_MEGA_THRESHOLDS
  if (verbose)
    cerr << "dsp::SKDetector::set_thresholds SKlimits(" << M << ", " << n_std_devs + 3 << ")" << endl;
  limits.set_std_devs(6);
  limits.calc_limits();

  mega_upper_thresh = (float) limits.get_upper_threshold();
  mega_lower_thresh = (float) limits.get_lower_threshold();
#endif

  if (verbose)
    cerr << "dsp::SKDetector::set_thresholds M=" << M << " n_std_devs="
         << n_std_devs  << " [" << lower_thresh << " - " << upper_thresh 
         << "]" << endl;
}

void dsp::SKDetector::set_channel_range (unsigned start, unsigned end)
{
  s_chan = start;
  e_chan = end;
}

void dsp::SKDetector::set_options (bool _disable_fscr, bool _disable_tscr,
                                   bool _disable_ft)
{
  disable_fscr = _disable_fscr;
  disable_tscr = _disable_tscr;
  disable_ft   = _disable_ft;
}

void dsp::SKDetector::set_input_tscr (TimeSeries * _input_tscr)
{
  input_tscr = _input_tscr;
}


void dsp::SKDetector::reserve()
{
  output->Observation::operator=(*input);
  output->set_npol (1);
  output->set_ndim (1);
  output->set_nbit (8);
  output->set_nchan (input->get_nchan());
  output->set_input_sample (input->get_input_sample());
  output->resize(input->get_ndat());
}

void dsp::SKDetector::transformation ()
{

  // ensure containers are large enough
  reserve();

  const float * indat  = input->get_dattfp();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  int64_t ndat         = input->get_ndat();

  unsigned char * outdat = output->get_datptr();

  if (e_chan == 0)
    e_chan = nchan;

  if (verbose || debugd < 1)
  {
    cerr << "dsp::SKDetector::transformation ndat= " << ndat 
         << " nchan=" << nchan << " nbit=" << input->get_nbit() 
         << " npol=" << npol << " ndim=" << input->get_ndim() << endl;

    cerr << "dsp::SKDetector::transformation OUTPUT ndat="
         << output->get_ndat() << " nchan=" << output->get_nchan()
         << " nbit=" << output->get_nbit() << " npol=" << output->get_npol() 
         << " ndim=" << output->get_ndim() << endl;
  }

  ndat_total += ndat * nchan;

  // reset the output to all 0 (no zapping)
  reset_mask();

  // apply the tscrunches SKFB estiamtes to the mask
  if (!disable_tscr)
    detect_tscr();

  // apply the SKFB estimates to the mask
  if (!disable_ft)
    detect_skfb();

  if (!disable_fscr)
    detect_fscr ();

  count_zapped ();

  if (debugd < 1)
    debugd++;
}

/*
 * Use the tscrunched SK statistic from the SKFB to detect RFI on eah channel
 */
void dsp::SKDetector::detect_tscr ()
{

  if (verbose)
    cerr << "dsp::SKDetector::detect_tscr()" << endl;

  const unsigned nchan   = input->get_nchan();
  const unsigned npol    = input->get_npol();
  int64_t ndat           = input->get_ndat();
  const float * indat    = input_tscr->get_dattfp();
  unsigned char * outdat = 0;

  float V_p0 = 0;
  float V_p1 = 0;

  if (ndat && (tscr_M != M * ndat))
  {
    tscr_M = M * ndat;

    if (verbose)
      cerr << "dsp::SKDetector::detect_tscr SKlimits(" << tscr_M << ", " << n_std_devs << ")" << endl;

    dsp::SKLimits limits(tscr_M, n_std_devs);
    limits.calc_limits();

    tscr_upper = (float) limits.get_upper_threshold();
    tscr_lower = (float) limits.get_lower_threshold();

    if (verbose)
      cerr << "dsp::SKDetector::detect_tscr M=" << tscr_M << " n_std_devs="
           << n_std_devs  << " [" << tscr_lower << " - " << tscr_upper
           << "]" << endl;
  }

  for (uint64_t ichan=s_chan; ichan < e_chan; ichan++)
  {
    // check the tscrunched value for this idat
    V_p0 = indat[2*ichan];
    V_p1 = indat[2*ichan+1];

    if ( V_p0 > tscr_upper ||
         V_p0 < tscr_lower ||
         V_p1 > tscr_upper ||
         V_p1 < tscr_lower )
    {
      if (verbose)
        cerr << "dsp::SKDetector::detect_tscr zap [" << V_p0 << ", " 
             << V_p1 << "] ichan=" << ichan << endl;
      outdat = output->get_datptr();
      for (unsigned idat=0; idat < ndat; idat++)
      {
        outdat[ichan] = 1;
        ndat_zapped_tscr++;
        outdat += nchan;
      }
    }
  }
}

void dsp::SKDetector::detect_skfb ()
{

  if (verbose)
    cerr << "dsp::SKDetector::detect_skfb()" << endl;

  const unsigned nchan   = input->get_nchan();
  const unsigned npol    = input->get_npol();
  int64_t ndat           = input->get_ndat();
  const float * indat    = input->get_dattfp();
  unsigned char * outdat = output->get_datptr();
#ifdef USE_MEGA_THRESHOLDS
  uint64_t zapped_mega = 0;
#endif
  float V_p0 = 0;
  float V_p1 = 0;

  // compare SK estimator for each pol to expected values
  for (uint64_t idat=0; idat < ndat; idat++)
  {
#ifdef USE_MEGA_THRESHOLDS
    zapped_mega = 0;
#endif

    // for each channel and pol in the SKFB
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      V_p0 =  indat[ichan*2];
      V_p1 =  indat[ichan*2+1];

      if ( V_p0 > upper_thresh || 
           V_p0 < lower_thresh ||
           V_p1 > upper_thresh || 
           V_p1 < lower_thresh )
      {
        outdat[ichan] = 1;

        // only count skfb zapped channels in the in-band region
        if (ichan > s_chan && ichan < e_chan)
          ndat_zapped_skfb++;

#ifdef USE_MEGA_THRESHOLDS
        if ( V_p0 > mega_upper_thresh || 
             V_p0 < mega_lower_thresh ||
             V_p1 > mega_upper_thresh || 
             V_p1 < mega_lower_thresh )
        {
          zapped_mega ++;
        }
#endif
      }
    }

#ifdef USE_MEGA_THRESHOLDS 
    if (zapped_mega > 10)
    {
      if (verbose)
        cerr << "ZAP mega n_bad_chan=" << zapped_mega << endl;
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        outdat[ichan] = 1;
      }
      ndat_zapped_mega += nchan;
    }
#endif
    
    indat += nchan * npol;
    outdat += nchan; 
  }
}

void dsp::SKDetector::reset_mask()
{
  unsigned nchan         = output->get_nchan();
  int64_t ndat           = output->get_ndat();
  unsigned char * outdat = output->get_datptr();

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (uint64_t idat=0; idat < ndat; idat++)
    {
      outdat[(idat*nchan) + ichan] = 0;
    }
  }
}



void dsp::SKDetector::count_zapped ()
{
  unsigned nchan         = output->get_nchan();
  int64_t ndat           = output->get_ndat();
  unsigned char * outdat = output->get_datptr();

  for (unsigned ichan=s_chan; ichan < e_chan; ichan++)
  {
    for (uint64_t idat=0; idat < ndat; idat++)
    {
      if (outdat[(idat*nchan) + ichan] == 1)
        ndat_zapped ++;
    }
  }
}

void dsp::SKDetector::detect_fscr ()
{ 
  if (verbose)
    cerr << "dsp::SKDetector::detect_fscr()" << endl;

  unsigned nchan       = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const int64_t ndat   = input->get_ndat();

  const float * indat  = input->get_dattfp();
  unsigned char * outdat = output->get_datptr();

  float sk_avg_p0 = 0;
  float sk_avg_p1 = 0;
  unsigned sk_avg_cnt = 0;
  
  // foreach SK integration
  for (uint64_t idat=0; idat < ndat; idat++)
  {
    sk_avg_p0 = 0;
    sk_avg_p1 = 0;
    sk_avg_cnt = 0;

    if (verbose)
      cerr << "dsp::SKDetector::detect_fscr idat=" << idat << endl;
    // accumulate the avg values for p0 and p1 
    for (unsigned ichan=s_chan; ichan < e_chan; ichan++)
    {
      if (outdat[ichan] == 0)
      {
        sk_avg_p0 += indat[ichan*2];
        sk_avg_p1 += indat[ichan*2+1];
        sk_avg_cnt++;
      }
    }

    if (sk_avg_cnt > 0)
    {
      sk_avg_p0 /= (float) sk_avg_cnt;
      sk_avg_p1 /= (float) sk_avg_cnt;

      float _M = (float) M;
      float mu2 = (4 * _M * _M) / ((_M-1) * (_M + 2) * (_M + 3));
      float one_sigma_idat = sqrt(mu2 / (float) sk_avg_cnt);

      float avg_upper_thresh = 1 + ((n_std_devs) * one_sigma_idat);
      float avg_lower_thresh = 1 - ((n_std_devs) * one_sigma_idat);

      if ((sk_avg_p0 > avg_upper_thresh) || (sk_avg_p0 < avg_lower_thresh) ||
          (sk_avg_p1 > avg_upper_thresh) || (sk_avg_p1 < avg_lower_thresh))
      {
        if (verbose)
          cerr << "Zapping all p0=" << sk_avg_p0
               << " p1=" << sk_avg_p1 << " [" << avg_lower_thresh << " - "
               << avg_upper_thresh << "] cnt=" << sk_avg_cnt << endl;
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          outdat[ichan] = 1;
        }
        ndat_zapped_fscr += sk_avg_cnt;
      }
    }

    indat += nchan * npol; 
    outdat += nchan;
  }
  
}
