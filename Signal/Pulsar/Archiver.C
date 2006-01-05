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
#include "Pulsar/Telescope.h"
#include "Pulsar/Receiver.h"

#include "Pulsar/FITSHdrExtension.h"

#include "Error.h"

#include <assert.h>
#include <time.h>

#ifdef sun
#include <ieeefp.h>
#endif

bool dsp::Archiver::verbose = false;

dsp::Archiver::Archiver ()
{
  archive_software = "Software Unknown";
  archive_dedispersed = false;
}

dsp::Archiver::~Archiver ()
{
}

void dsp::Archiver::set_archive_class (const string& class_name)
{
  archive_class_name = class_name;
}

void dsp::Archiver::set_archive (Pulsar::Archive* archive)
{
  single_archive = archive;
}

//! Add a Pulsar::Archive::Extension to those added to the output archive
void dsp::Archiver::add_extension (Pulsar::Archive::Extension* extension)
{
  extensions.push_back (extension);
}

//! Set the Response from which Passband Extension will be constructed
void dsp::Archiver::set_passband (const Response* _passband)
{
  passband = _passband;
}

//! Set the Operation instances for the dspReduction Extension
void dsp::Archiver::set_operations (const vector<Operation*>& ops)
{
  if (verbose) cerr << "dsp::Archiver::set_operations " << ops.size() 
                    << " operations" << endl;
 
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

  if (!single_archive) {
    cerr << "dsp::Archiver::unload archive '"
	 << archive->get_filename() << "'" << endl;
    
    archive -> unload();
  }

}

void dsp::Archiver::add (Pulsar::Archive* archive, const PhaseSeries* phase)
try {

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

void dsp::Archiver::set (Pulsar::Archive* archive, const PhaseSeries* phase)
try {

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
  unsigned nsub = 1;

  unsigned effective_npol = npol;

  if ( phase->get_domain() == "Lag" )
    nsub = ndim;
  else
    effective_npol *= ndim;

  if( verbose )
    cerr << "dsp::Archiver::set Pulsar::Archive nsub=" << nsub 
	 << " npol=" << effective_npol << " nchan=" << nchan 
	 << " nbin=" << nbin << endl;

  archive-> resize (nsub, effective_npol, nchan, nbin);

  Pulsar::FITSHdrExtension* ext;
  ext = archive->get<Pulsar::FITSHdrExtension>();
  
  if (ext) {

    // Make sure the start time is aligned with pulse phase zero
    // as this is what the PSRFITS format expects.

    MJD initial = phase->get_start_time();

    Phase inphs = phase->get_folding_polyco()->phase(initial);

    double dtime = inphs.fracturns() * phase->get_folding_period();
    initial -= dtime;

    ext->start_time = initial;

    // In keeping with tradition, I'll set this to a value that should
    // work in most places for the next 50 years or so ;)

    ext->set_coord_mode("J2000");

    // Set the ASCII date stamp from the system clock (in UTC)

    time_t thetime;
    time(&thetime);
    string time_str = asctime(gmtime(&thetime));

    // Cut off the line feed character
    time_str = time_str.substr(0,time_str.length() - 1);

    ext->set_date_str(time_str);
  }
  
  /*! Install the given ephemeris and calls update_model
  archive-> set_ephemeris (const psrephem& ephemeris);
  */

  archive-> set_telescope_code ( phase->get_telescope_code() );

  archive-> set_type ( phase->get_type() );
  if (phase->get_state() == Signal::NthPower) {
    fprintf(stderr, "Pulsar::Archiver:set State is NthPower - setting Archive state to Intensity\n");
    archive->set_state (Signal::Intensity);
  }
  else {
    archive-> set_state ( phase->get_state() );
  }

  archive-> set_scale ( Signal::FluxDensity );

  archive-> set_source ( phase->get_source() );
  archive-> set_coordinates ( phase->get_coordinates() );
  archive-> set_bandwidth ( phase->get_bandwidth() );
  archive-> set_centre_frequency ( phase->get_centre_frequency() );
  archive-> set_dispersion_measure ( phase->get_dispersion_measure() );

  archive-> set_faraday_corrected (false);
  archive-> set_dedispersed( archive_dedispersed );

  for (unsigned isub=0; isub < nsub; isub++)
    set (archive->get_Integration(isub), phase, isub, nsub);

  // set any available extensions
  Pulsar::dspReduction* dspR = archive -> getadd<Pulsar::dspReduction>();
  if (dspR){
    set (dspR);
    dspR->set_name( phase->get_machine() );
  }

  Pulsar::TwoBitStats* tbc = archive -> getadd<Pulsar::TwoBitStats>();
  if (tbc)
    set (tbc);

  Pulsar::Passband* pband = archive -> getadd<Pulsar::Passband>();
  if (pband)
    set (pband);

  Pulsar::Telescope* telescope = archive -> getadd<Pulsar::Telescope>();
  telescope->set_coordinates (phase -> get_telescope_code());

  // default Receiver extension
  archive -> getadd<Pulsar::Receiver>();

  for (unsigned iext=0; iext < extensions.size(); iext++)
    archive -> add_extension ( extensions[iext] );

  // dsp::PhaseSeries has either (both eph and polyco) or (none)
  // set_model must be called after the Integration::MJD has been set
  if( phase->get_pulsar_ephemeris() ){
    archive-> set_model ( *(phase->get_folding_polyco()) );
    archive-> set_ephemeris( *(phase->get_pulsar_ephemeris()), false );
  }

  archive-> set_filename (get_filename (phase));

  if (verbose) cerr << "dsp::Archiver set archive filename to '"
		    << archive->get_filename() << "'" << endl;

}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Archive";
}


void dsp::Archiver::set (Pulsar::Integration* integration,
			 const PhaseSeries* phase,
			 unsigned isub, unsigned nsub)
try {

  if (verbose)
    cerr << "dsp::Archiver::set Pulsar::Integration" << endl;

  unsigned npol = phase->get_npol();
  unsigned nchan = phase->get_nchan();
  unsigned ndim = phase->get_ndim();

  unsigned effective_npol = npol;

  if (nsub > 1) {
    if (ndim != nsub)
      cerr << "YIKES!" << endl;
    ndim = 1;
  }
  else
    effective_npol *= ndim;

  if ( integration->get_npol() != effective_npol)
    throw Error (InvalidParam, "dsp::Archiver::set (Pulsar::Integration)",
                 "Integration.npol=%d != PhaseSeries.npol=%d", 
                 integration->get_npol(), effective_npol);

  if ( integration->get_nchan() != nchan)
    throw Error (InvalidParam, "dsp::Archiver::set (Pulsar::Integration)",
                 "Integration.nchan=%d != PhaseSeries.nchan=%d", 
                 integration->get_nchan(), nchan);

  integration-> set_epoch ( phase->get_mid_time() );
  integration-> set_duration ( phase->get_integration_length() );

  integration-> set_folding_period ( phase->get_folding_period () );

  unsigned offchan = 0;
  if ( phase->get_swap() )
    offchan = nchan/2; // swap the channels (passband re-order)

  for (unsigned ichan=0; ichan<nchan; ichan++)
    for (unsigned ipol=0; ipol<npol; ipol++)
      for (unsigned idim=0; idim<ndim; idim++) {

	unsigned poln = ipol*ndim+idim;
	unsigned chan = (ichan+offchan)%nchan;

	if (nsub > 1)
	  idim = isub;

	Pulsar::Profile* profile = integration->get_Profile(poln, chan);

	if( verbose )
	  cerr << "dsp::Archiver::set Pulsar::Integration ipol=" << poln
	       << " ichan=" << chan << " nbin=" << profile->get_nbin() << endl;

	set (profile, phase, ichan, ipol, idim);

      }
}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Integration";
}



void dsp::Archiver::set (Pulsar::Profile* profile,
			 const PhaseSeries* phase,
			 unsigned ichan, unsigned ipol, unsigned idim)
try {

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

  if (scale == 0 || !finite(scale))
    throw Error (InvalidParam, string(), "invalid scale=%lf", scale);

  for (unsigned ibin = 0; ibin<nbin; ibin++) {

    if (phase->get_hit(ibin) == 0) {

      zeroes ++;
      *to = 0.0;

    }
    else {

      if (!finite(*from))
	throw Error (InvalidParam, string(),
		     "invalid data[ichan=%d][ipol=%d][idim=%d][ibin=%d]=%f",
		     ichan, ipol, idim, ibin, *from);

      *to = *from / (scale * double( phase->get_hit(ibin) ));

    }

    to ++;
    from += ndim;
  }

  if (zeroes && verbose)
    cerr << "dsp::Archiver::set Pulsar::Profile Warning: " << zeroes 
	 << " out of " << nbin << " bins with zero hits\r";
}
catch (Error& error) {
  throw error += "dsp::Archiver::set Pulsar::Profile";
}

