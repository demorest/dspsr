#include "polyco.h"
#include "psrephem.h"

#include "dsp/Archiver.h"
#include "dsp/PhaseSeries.h"
#include "dsp/Response.h"
#include "dsp/Operation.h"
#include "dsp/TwoBitCorrection.h"

#include "Pulsar/Archive.h"
#include "Pulsar/Integration.h"
#include "Pulsar/Profile.h"

#include "Pulsar/dspReduction.h"
#include "Pulsar/TwoBitStats.h"
#include "Pulsar/Passband.h"

#include "Error.h"

#include <assert.h>

bool dsp::Archiver::verbose = false;

dsp::Archiver::Archiver ()
{
}

dsp::Archiver::~Archiver ()
{
}

void dsp::Archiver::set_archive_class (const char* _archive_class_name)
{
  archive_class_name = _archive_class_name;
}

void dsp::Archiver::set_archive (Pulsar::Archive* archive)
{
  single_archive = archive;
}

//! Set the Response from which Passband Extension will be constructed
void dsp::Archiver::set_passband (const Response* _passband)
{
  passband = _passband;
}

//! Set the Operation instances for the dspReduction Extension
void dsp::Archiver::set_operations (const vector<Operation*>& ops)
{
  operations.resize ( ops.size() );
  for (unsigned iop=0; iop < ops.size(); iop++)
    operations[iop] = ops[iop];
}

void dsp::Archiver::unload ()
{
  if (!single_archive && archive_class_name.size() == 0)
    throw Error (InvalidState, "dsp::Archiver::unload", 
		 "neither Archive nor class name specified");

  if (!profiles)
    throw Error (InvalidState, "dsp::Archiver::unload",
		 "Profile data not provided");

  Reference::To<Pulsar::Archive> archive;

  if (single_archive) {
    // refer to the single archive to which all sub-integration will be written
    archive = single_archive;
    // add the main data
    add (archive, profiles);
  }
  else {
    // create a new archive
    archive = Pulsar::Archive::new_Archive (archive_class_name);
    // set the main data
    set (archive, profiles);
  }

  // set any available extensions

  unsigned next = archive->get_nextension ();
  Pulsar::Archive::Extension* ext = 0;

  for (unsigned iext=0; iext<next; iext++)  {
    ext=const_cast<Pulsar::Archive::Extension*>(archive->get_extension(iext));

    Pulsar::dspReduction* dspR = dynamic_cast<Pulsar::dspReduction*>( ext );
    if (dspR)
      set (dspR);

    Pulsar::TwoBitStats* tbc = dynamic_cast<Pulsar::TwoBitStats*>( ext );
    if (tbc)
      set (tbc);

    Pulsar::Passband* pband = dynamic_cast<Pulsar::Passband*>( ext );
    if (pband)
      set (pband);
  }

  if (!single_archive) {
    cerr << "dsp::Archiver::unload archive '"
	 << archive->get_filename() << "'" << endl;
  
    archive -> unload();
  }

}

void dsp::Archiver::add (Pulsar::Archive* archive, const PhaseSeries* phase)
{ try {

  if (verbose)
    cerr << "dsp::Archiver::add Pulsar::Archive" << endl;

  if (!archive)
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
		 "no Archive");

  if (!phase) 
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
		 "no PhaseSeries");

  unsigned nsub = archive->get_nsubint();

  if (!nsub) {
    set (archive, phase);
    return;
  }

  unsigned npol = phase->get_npol() * phase->get_ndim();

  // simple sanity check
  if (archive->get_npol() != npol)
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
		 "Pulsar::Archive::npol=%d != PhaseSeries::npol=%d",
		 archive->get_npol(), npol);
  if (archive->get_nchan() != phase->get_nchan())
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
		 "Pulsar::Archive::nchan=%d != PhaseSeries::nchan=%d",
		 archive->get_nchan(), phase->get_nchan());
  if (archive->get_nbin() != phase->get_nbin())
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
		 "Pulsar::Archive::nbin=%d != PhaseSeries::nbin=%d",
		 archive->get_nbin(), phase->get_nbin());
  
  archive-> resize (nsub + 1);
  set (archive-> get_Integration(nsub), phase);

}
catch (Error& error) {
  throw error += "dsp::Archiver::add Pulsar::Archive";
}
}

void dsp::Archiver::set (Pulsar::Archive* archive, const PhaseSeries* phase)
{ try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::Archive" << endl;

  if (!archive)
    throw Error (InvalidParam, "dsp::Archiver::set Pulsar::Archive",
		 "no Archive");

  if (!phase)
    throw Error (InvalidParam, "dsp::Archiver::set Pulsar::Archive",
		 "no PhaseSeries");

  unsigned npol = phase->get_npol();
  unsigned nchan = phase->get_nchan();
  unsigned nbin = phase->get_nbin();
  unsigned ndim = phase->get_ndim();

  archive-> resize (1, npol*ndim, nchan, nbin);

  /*! Install the given ephemeris and calls update_model
  archive-> set_ephemeris (const psrephem& ephemeris);
  */

  archive-> set_telescope_code ( phase->get_telescope_code() );

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

  // dsp::PhaseSeries has either (both eph and polyco) or (none)
  // set_model must be called after the Integration::MJD has been set
  if( phase->get_pulsar_ephemeris() ){
    archive-> set_model ( *(phase->get_folding_polyco()) );
    archive-> set_ephemeris( *(phase->get_pulsar_ephemeris()), false );
  }

  archive-> set_filename (get_filename (phase));

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

  if ( integration->get_npol() != npol*ndim)
    throw Error (InvalidParam, "dsp::Archiver::set (Pulsar::Integration)",
                 "Integration.npol=%d != PhaseSeries.npol=%d", 
                 integration->get_npol(), npol*ndim);

  if ( integration->get_nchan() != nchan)
    throw Error (InvalidParam, "dsp::Archiver::set (Pulsar::Integration)",
                 "Integration.nchan=%d != PhaseSeries.nchan=%d", 
                 integration->get_nchan(), nchan);

  integration-> set_epoch ( phase->get_mid_time() );
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
	 << " out of " << nbin << " bins with zero hits\r";
}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Profile";
}}
