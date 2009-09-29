/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*

  This program corrects the 4-sample swap bug introduced to the 
  LBADR_TwoBitCorrection class on 2008/05/21 and later fixed on
  2009/07/31.

  The bug resulted in swapping the order of the 2-bit numbers in each
  byte; i.e. 0,1,2,3 -> 3,2,1,0

  The problem is modeled using 4 separate under-sampled and offset
  sampling functions which are shifted by +3,+1,-1,-3 samples before
  being recombined.

  Under-sampling by a factor of four divides the band in two.  Each
  half contains the sum of

  - the original band in this half
  - the other half of the band
  - the Hermitian transpose [F(-w)=F*(w)] of the original band in this half
  - the Hermitian transpose of the other half of the band

  Each component is added after multiplication by a different complex
  phase factor in each half of the band.

  The correction is performed by noting that sets of four corrupted
  channels are simply linear combinations of the original four
  channels.  A 4x4 matrix representing this linear transformation is
  inverted and used to recover the original four channels.  Noise is
  amplified in the process, but this version of this code does nothing
  to compensate for this effect.

*/

#include "Pulsar/Application.h"
#include "Pulsar/UnloadOptions.h"

#include "Pulsar/Archive.h"
#include "Pulsar/Integration.h"
#include "Pulsar/Profile.h"

#include "Matrix.h"

#include <iostream>

using namespace std;
using namespace Pulsar;

//! Pulsar Archive Editing application
class lbadr64_fix: public Pulsar::Application
{
public:

  //! Default constructor
  lbadr64_fix ();

  //! Process the given archive
  void process (Pulsar::Archive*);

protected:

  //! Add command line options
  void add_options (CommandLine::Menu&) {}

  //! commands to be executed
  vector< Matrix<4,4,double> > fix;
  void build ();
};

int main (int argc, char** argv)
{
  lbadr64_fix program;
  return program.main (argc, argv);
}

lbadr64_fix::lbadr64_fix ()
  : Pulsar::Application ("lbadr64_fix", "corrects LBADR 2-bit unpacker bug")
{
  version = "$Id: lbadr64_fix.C,v 1.3 2009/09/29 07:19:12 straten Exp $";
  add( new Pulsar::UnloadOptions );
}


typedef complex<double> cplex;

/*
  Computes the spectral representation of the convolution as a function
  of phi, which varies linearly with frequency.
*/
cplex spectral (const cplex& p0,  // first half of band, positive frequencies
		const cplex& p1,  // second half of band, positive frequencies
		const cplex& n1,  // second half, negative [F(-w)=F*(w)]
		const cplex& n0,  // first half, negative 
		double phi)
{
  cplex i (0,1);

  // picket fence delayed by zero
  cplex x0 =   n1 +   n0 + p0 +   p1;
  // picket fence delayed by one
  cplex x1 = - n1 - i*n0 + p0 + i*p1;
  // picket fence delayed by two
  cplex x2 =   n1 -   n0 + p0 -   p1;
  // picket fence delayed by three
  cplex x3 = - n1 + i*n0 + p0 - i*p1;

#if 1
  x0 = conj(x0);
  x1 = conj(x1);
  x2 = conj(x2);
  x3 = conj(x3);
#endif

  // shift by one sample
  cplex e1( cos(phi), sin(phi) );
  // shift by three samples
  cplex e3( cos(3*phi), sin(3*phi) );

  // result is sum of all picket fenced and shifted terms
  return x0*e3 + x1*e1 + x2*conj(e1) + x3*conj(e3);
}

void lbadr64_fix::build ()
{
  const unsigned nchan = fix.size();

  const unsigned NCHAN = 4096;
  const unsigned ratio = NCHAN / nchan;

  Matrix<4,4,double> total;
  unsigned jchan = 0;
  unsigned count = 0;

  for (unsigned ichan=0; ichan < NCHAN/4; ichan++)
  {
    unsigned chan[4] = { 
      /* p0 */ ichan,
      /* p1 */ NCHAN/2 + ichan,
      /* n1 */ NCHAN - ichan - 1,
      /* n0 */ NCHAN/2 - ichan -1
    };

    double phase[4];

    for (unsigned i=0; i < 4; i++)
      phase[i] = chan[i] * M_PI / NCHAN;
    
    Matrix<4,4,double> corruption;
    cplex zero (0.0, 0.0);
    cplex r;

    for (unsigned i=0; i<4; i++)
    {
      vector<cplex> p (4, zero);  // the four inputs
      p[i] = cplex (1.0, 1.0);
      
      r = spectral (p[0], p[1], p[2], p[3], phase[0]);
      corruption[0][i] = norm(r);
      
      r = spectral (p[1], p[2], p[3], p[0], phase[1]);
      corruption[1][i] = norm(r);
      
      r = spectral (p[2], p[1], p[0], p[3], phase[2]);
      corruption[2][i] = norm(r);
      
      r = spectral (p[3], p[2], p[1], p[0], phase[3]);
      corruption[3][i] = norm(r);
    }

    total += corruption;
    count ++;

    if (count == ratio)
    {
      total /= count;
      // cerr << jchan << endl << total << endl;

      fix[jchan] = inv (total);

      jchan++;
      count = 0;
    }
  }
}

void lbadr64_fix::process (Pulsar::Archive* archive)
{
  const unsigned nsubint = archive->get_nsubint();
  const unsigned nchan = archive->get_nchan();
  const unsigned npol = archive->get_npol();
  const unsigned nbin = archive->get_nbin();

  if (fix.size() != nchan)
  {
    fix.resize (nchan);
    build();
  }

  for (unsigned isubint=0; isubint < nsubint; isubint++)
  {
    Integration* subint = archive->get_Integration (isubint);

    for (unsigned ichan=0; ichan < nchan/4; ichan++)
    {
      unsigned fix_chan[4] = { 
	/* p0 */ ichan,
	/* p1 */ nchan/2 + ichan,
	/* n1 */ nchan - ichan - 1,
	/* n0 */ nchan/2 - ichan -1
      };

      for (unsigned ipol=0; ipol < npol; ipol++)
      {
	Profile* prof[4];

	for (unsigned i=0; i < 4; i++)
	  prof[i] = subint->get_Profile (ipol, fix_chan[i]);

	for (unsigned ibin=0; ibin < nbin; ibin++)
	{
	  Vector<4,double> tofix;
	  for (unsigned i=0; i<4; i++)
	    tofix[i] = prof[i]->get_amps()[ibin];

	  Vector<4,double> fixed = fix[ichan] * tofix;
	    
	  for (unsigned i=0; i<4; i++)
	    prof[i]->get_amps()[ibin] = fixed[i];
	}
      }
    }
  }
}
