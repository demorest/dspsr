#include "polyco.h"

#include "dsp/Archiver.h"
#include "dsp/PhaseSeries.h"

#include "Pulsar/Archive.h"
#include "Pulsar/Integration.h"
#include "Pulsar/Profile.h"
#include "Error.h"

bool dsp::Archiver::verbose = false;

void dsp::Archiver::set (Pulsar::Archive* archive, const PhaseSeries* phase)
{ try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::Archive" << endl;

  unsigned npol = phase->get_npol();
  unsigned nchan = phase->get_nchan();
  unsigned nbin = phase->get_nbin();
  unsigned ndim = phase->get_ndim();

  archive-> resize (1, npol*ndim, nchan, nbin);

  /*! Install the given ephemeris and calls update_model
  archive-> set_ephemeris (const psrephem& ephemeris);
  */

  archive-> set_telescope_code ( phase->get_telescope() );

  archive-> set_type ( phase->get_type() );
  archive-> set_basis ( phase->get_basis() );
  archive-> set_state ( phase->get_state() );

  archive-> set_source ( phase->get_source() );
  archive-> set_coordinates ( phase->get_coordinates() );
  archive-> set_bandwidth ( phase->get_bandwidth() );
  archive-> set_centre_frequency ( phase->get_centre_frequency() );
  archive-> set_dispersion_measure ( phase->get_dispersion_measure() );

  archive-> set_flux_calibrated (false);
  archive-> set_feedangle_corrected (false);
  archive-> set_iono_rm_corrected (false);
  archive-> set_ism_rm_corrected (false);
  archive-> set_parallactic_corrected (false);

  set (archive-> get_Integration(0), phase);

  // set_model must be called after the Integration::MJD has been set
  if( !phase->get_folding_polyco() ){
    fprintf(stderr,"using no polyco whatsoever\n");
    //archive->set_model( polyco(phase->get_mid_time(),
    //		       phase->get_dispersion_measure(),
    //		       1.0/phase->get_folding_period(),
    //		       phase->get_telescope()) );
  }
  else{
    archive-> set_model ( *(phase->get_folding_polyco()) );
  }

  archive-> set_filename (phase->get_default_id () + ".ar");
}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Archive";
}
}

void dsp::Archiver::set (Pulsar::Integration* integration,
			 const PhaseSeries* phase)
{ try {
  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::Integration" << endl;

  unsigned npol = phase->get_npol();
  unsigned nchan = phase->get_nchan();
  unsigned ndim = phase->get_ndim();

  integration-> resize (npol*ndim, nchan);

  integration-> set_mid_time ( phase->get_mid_time() );
  integration-> set_duration ( phase->get_integration_length() );

  integration-> set_centre_frequency ( phase->get_centre_frequency() );
  integration-> set_bandwidth ( phase->get_bandwidth() );
  integration-> set_dispersion_measure ( phase->get_dispersion_measure() );
    
  integration-> set_folding_period ( phase->get_folding_period () );
  integration-> set_basis ( phase->get_basis() );
  integration-> set_state ( phase->get_state() );

  unsigned offchan = 0;
  if ( phase->get_swap() )
    offchan = nchan/2; // swap the channels (passband re-order)

  for (unsigned ichan=0; ichan<nchan; ichan++)
    for (unsigned ipol=0; ipol<npol; ipol++)
      for (unsigned idim=0; idim<ndim; idim++) {
	unsigned poln = ipol*ndim+idim;
	unsigned chan = (ichan+offchan)%nchan;
	set (integration->get_Profile(poln, chan),
		     phase, ichan, ipol, idim);
      }
}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Integration";
}
}


void dsp::Archiver::set (Pulsar::Profile* profile,
				 const PhaseSeries* phase,
				 unsigned ichan, unsigned ipol, unsigned idim)
{ try {
  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::Profile"
      " ichan=" << ichan << " ipol=" << ipol << " idim=" << idim << "\r";

  unsigned nbin = phase->get_nbin();
  unsigned npol = phase->get_npol();
  unsigned ndim = phase->get_ndim();

  assert (ipol < npol);
  assert (idim < ndim);

  profile-> resize (nbin);
  profile-> set_centre_frequency (phase->get_centre_frequency (ichan));
  profile-> set_weight (1.0);

  const float* from = phase->get_datptr (ichan, ipol) + idim;
  float* to = profile->get_amps ();

  unsigned zeroes = 0;

  double scale = phase->get_scale ();

  if (scale == 0 || isnan(scale))
    throw Error (InvalidParam, "dsp::Archiver::set Pulsar::Profile",
		"invalid scale=%lf", scale);

  for (unsigned ibin = 0; ibin<nbin; ibin++) {

    if (phase->get_hit(ibin) == 0) {
      zeroes ++;
      *to = 0.0;
    }
    else
      *to = *from / (scale * double( phase->get_hit(ibin) ));

    to ++;
    from += ndim;
  }

  if (zeroes)
    cerr << "dsp::Archiver::set Pulsar::Profile Warning: " << zeroes 
	 << " bins with zero hits!" << "\r";
}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Profile";
}}

