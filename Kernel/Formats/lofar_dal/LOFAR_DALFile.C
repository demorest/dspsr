/***************************************************************************
 *
 *   Copyright (C) 2005-2008, 2013 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 *   Change Log:
 *   2013 Jul 26  James M Anderson  changes to get lofar_dal to work with the
 *                current LOFAR DAL version 2.5.0
 *   2014 Nov 10  Vlad Kondratiev   bug fixes, improved functionality 
 *
 ***************************************************************************/

using namespace std;

#include "dsp/LOFAR_DALFile.h"

#include "dal/lofar/BF_File.h"
#include <iomanip>  // for setprecision use
using namespace dal;

class dsp::LOFAR_DALFile::Handle
{
public:
  std::vector< BF_File* > bf_file;
  std::vector< BF_StokesDataset* > bf_stokes;
};

dsp::LOFAR_DALFile::LOFAR_DALFile (const char* filename) : File ("LOFAR_DAL")
{
  handle = 0;
}

dsp::LOFAR_DALFile::~LOFAR_DALFile ( )
{
  close ();
}

bool dsp::LOFAR_DALFile::is_valid (const char* filename) const try
{
  BF_File* bf_file = new BF_File (filename);

  // memory leak
  // for some reason, deleting the file causes a segfault
  // delete bf_file;

  return true;
}
catch (HDF5Exception& error)
{
  if (verbose)
    cerr << "dsp::LOFAR_DALFile::is_valid exception " << error.what() << endl;
  return false;
}

void dsp::LOFAR_DALFile::open_file (const char* filename)
{

  BF_File* bf_file = new BF_File (filename);

  Attribute<std::string> PI = bf_file->projectPI();
  if (verbose) {
   if (PI.exists())
     cerr << "LOFAR_DALFile::open_file PI=" << PI.get() << endl;
  }

  Attribute<std::string> projectContact = bf_file->projectContact();
  if (verbose) {
   if (projectContact.exists())
     cerr << "LOFAR_DALFile::open_file PROJECT_CONTACT=" << projectContact.get() << endl;
  }

  Attribute< std::vector<std::string> > BFtargets = bf_file->targets();
  if (verbose) {
   if (BFtargets.exists())
     {
      std::vector<std::string> t = BFtargets.get();
      std::vector<std::string>::size_type i=0;
      for(std::vector<std::string>::iterator it = t.begin();
          it != t.end(); ++it, i++)
      {
        cerr << "LOFAR_DALFile::open_file TARGET" << i << "=" << *it << endl;
      }
     }
   else
     cerr << "TARGET does not exist" << endl;
  }

  // getting frequency center
  Attribute<double> freq = bf_file->observationFrequencyCenter();
  if (!freq.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "observationFrequencyCenter not defined");
  else if (verbose) cerr << "LOFAR_DALFile::open_file observation frequency=" << setprecision(20) << freq.get() << endl;

  // getting number of SAPs
  Attribute<unsigned> nsap = bf_file->observationNofSubArrayPointings();
  if (!nsap.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "observationNofSubArrayPointings not defined");
  else if (verbose) cerr << "LOFAR_DALFile::open_file number of SAPs=" << nsap.get() << endl;

  // getting  the instance of SAP
  // checking all SAPs if they exist to pick the right one (if there will be two SAPs in one file, only
  // the first one will be picked up)
  unsigned sap_index;
  for (sap_index=0; sap_index<nsap.get(); sap_index++) {
    if (bf_file->subArrayPointing(sap_index).exists()) break;
  }
  BF_SubArrayPointing sap = bf_file->subArrayPointing(sap_index);

  Attribute<unsigned> nbeam = sap.observationNofBeams();
  if (!nbeam.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "sap.observationNofBeams not defined");
  else if (verbose) cerr << "LOFAR_DALFile::open_file number of beams=" << nbeam.get() << endl;

  // getting the instance of first TA beam in the SAP
  // checking all TABs in the SAP if they exist in the file until the first one that exists is found
  unsigned tab_index;
  for (tab_index=0; tab_index<nbeam.get(); tab_index++) {
    if (sap.beam(tab_index).exists()) break;
  }
  BF_BeamGroup beam = sap.beam(tab_index);

  // getting the center frequency of the beam
  Attribute<double> freq2 = beam.beamFrequencyCenter();
  if (!freq2.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "beam.beamFrequencyCenter not defined");
  else if (verbose) cerr << "LOFAR_DALFile::open_file beam frequency=" << setprecision(20) << freq2.get() << endl;

  // getting the subband width
  Attribute<double> bw2 = beam.subbandWidth();
  if (!bw2.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "beam.subbandWidth not defined");
  else if (verbose) cerr << "LOFAR_DALFile::open_file sap subbandwidth=" << setprecision(20) << bw2.get() << endl;

  // getting number of channels per subband
  Attribute<unsigned> nchan = beam.channelsPerSubband();
  if (!nchan.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "beam.channelsPerSubband not defined");
  else if (verbose) cerr << "LOFAR_DALFile::open_file number of channels=" << nchan.get() << endl;

  // getting the pointer for the Stokes class
  BF_StokesDataset* stokes = 0;

  for (unsigned i=0; i<4; i++) {
    BF_StokesDataset tmp = beam.stokes(i);
    if (tmp.exists())
      stokes = new BF_StokesDataset (beam.stokes(i));
  }

  // getting the Stokes component
  Attribute<std::string> stokesC = stokes->stokesComponent();
  if (verbose) if (stokesC.exists()) cerr << "stokes component=" << stokesC.get() << endl;

  // getting the number of subbands
  Attribute<unsigned> nsub = stokes->nofSubbands();
  if (verbose) {
    if (nsub.exists())
      cerr << "nsub=" << nsub.get() << endl;
    else cerr << "stokes nofSubbands not defined" << endl;
  }

  // getting the number of channels for each subband
  Attribute< std::vector<unsigned> > nofchan = stokes->nofChannels();
  if (verbose) {
    if (nchan.exists())
    {
      std::vector<unsigned> nchan = nofchan.get();
      cerr << "stokes nofChannels size=" << nchan.size() << endl;
      for (unsigned i=0; i<nchan.size(); i++)
        cerr << "stokes nofChannels[" << i << "]=" << nchan[i] << endl;
    } else cerr << "stokes nofChannels not defined" << endl;
  }

  // getting the rank of the dataset
  size_t ndim= stokes->ndims();
  if (verbose) cerr << "stokes ndim=" << ndim << endl;

  if (verbose) {
   std::vector<std::string> files = stokes->externalFiles();
   for (unsigned i=0; i<files.size(); i++)
     cerr << "files[" << i << "]=" << files[i] << endl;
  }

  /* **********************************************************************
   *
   *
   *
   *
   * ********************************************************************** */

  // set Observation attributes

  // getting telescope
  Attribute<std::string> telescope = bf_file->telescope();
  if (telescope.exists()) {
    if (verbose) cerr << "LOFAR_DALFile::open_file telescope=" << telescope.get() << endl;
    // setting the telescope
    get_info()->set_telescope(telescope.get());
  }

  // setting machine
  get_info()->set_machine("COBALT");

  // getting the vector of targets
  Attribute< std::vector<std::string> > targets = beam.targets();
  if (targets.exists()) {
    std::vector<std::string> t = targets.get();
    if (t.size() != 0) {
     get_info()->set_source(t.front());
     if (verbose) cerr << "target = " << t.front() << endl;
    } else { if (verbose) cerr << "targets vector is empty" << endl; }
  } else { if (verbose) cerr << "beam target does not exist" << endl; }

  // getting number of samples
  Attribute<unsigned> nsamp = stokes->nofSamples();
  if (nsamp.exists())
    get_info()->set_ndat( nsamp.get() );

  // are data in Complex Voltage format?
  Attribute<bool> volts = beam.complexVoltage();
  if (volts.exists() && volts.get() == 1) {
    if (ndim != 2 && verbose) {
     cerr << "Error: The rank of the dataset (" << ndim << ") does not correspond to Complex Voltage data!" << endl;
    }
    get_info()->set_ndim (2);
  }
  else {
    if (ndim != 1) {
     if (verbose) cerr << "Warning! The rank of the dataset (" << ndim << ") is larger than returned by 'complexVoltage' function." << endl;
     get_info()->set_ndim (ndim);
    } else {
     get_info()->set_ndim (1);
    }
  }
 
  // check for which coordinate is Spectral
  unsigned spectral_dim = 1;

  // getting instance of Coordinates container
  CoordinatesGroup coord = beam.coordinates();
  if (coord.exists())
  {
    Attribute< std::vector<std::string> > types = coord.coordinateTypes();
    if (types.exists())
    {
      std::vector<std::string> t = types.get();
      for (unsigned i=0; i<t.size(); i++)
      {
	if (t[i] == "Spectral")
	{
	  spectral_dim = i;
	  break;
	}
      }
    }
  }

  std::vector<ssize_t> dims = stokes->dims();
  get_info()->set_nchan( dims[spectral_dim] );
  
  // getting number of Stokes components in one file
  //Attribute<unsigned> npol = beam.nofStokes();
  // getting number of Stokes components in the observation
  Attribute<unsigned> npol = beam.observationNofStokes();
  unsigned stokes_npol = 1;

  if (npol.exists())
    stokes_npol = npol.get();

  if (stokes_npol == 1)
  {
    get_info()->set_npol (1);
    get_info()->set_state( Signal::Intensity );
  }
  else
  {
    if (get_info()->get_ndim() == 2)  // complexVoltages == true
    {
      get_info()->set_npol (2);
      get_info()->set_state( Signal::Analytic );
    }
    else
    {
      get_info()->set_npol (4);
      get_info()->set_state( Signal::Stokes );
    }
  }

  get_info()->set_nbit (32);

  // getting split Frequency center of the beam
  Attribute<double> cfreq = beam.beamFrequencyCenter();
  if (!cfreq.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "beamFrequencyCenter not defined");
  // assuming cfreq is in MHz (default?)
  get_info()->set_centre_frequency( cfreq.get() /** 1e-6 */ );

  get_info()->set_dc_centred( false );

  // getting the start MJD
  Attribute<double> mjd = bf_file->observationStartMJD();
  if (mjd.exists())
    get_info()->set_start_time( MJD(mjd.get()) );
    if (verbose) cerr << "MJD=" << setprecision(20) << get_info()->get_start_time() << endl;

  // getting the clock rate
  Attribute<double> cRate = bf_file->clockFrequency();
  if (verbose) {
    if (cRate.exists())
      cerr << "clockRate=" << setprecision(20) << cRate.get() << endl;
    else cerr << "clockRate undefined" << endl;
  }

  // getting the sampling rate
  Attribute<double> sRate = beam.samplingRate();
  if (verbose) {
    if (sRate.exists())
      cerr << "samplingRate=" << setprecision(20) << sRate.get() << endl;
    else cerr << "samplingRate undefined" << endl;
  }

  // getting the sampling time
  Attribute<double> sTime = beam.samplingTime();
  if (sTime.exists()) {
    if (verbose) cerr << "samplingTime=" << setprecision(20) << sTime.get() << endl;
    get_info()->set_rate (1./sTime.get());
    if (verbose) cerr << "[ set rate to " << setprecision(20) << 1./sTime.get() << " Hz]" << endl;
  }
  else if (verbose) cerr << "samplingTime undefined" << endl;

  // getting the channel width
  Attribute<double> rate = beam.channelWidth();
  if (!rate.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file", "beam.channelWidth not defined");
  else if (verbose) cerr << "channel Width=" << setprecision(20) << rate.get() << " Hz" << endl;

  // getting the subband width
  Attribute<double> subwidth = beam.subbandWidth();
  if (verbose) {
    if (subwidth.exists())
      cerr << "subband Width=" << setprecision(20) << subwidth.get() << " Hz" << endl;
    else cerr << "subband Width undefined" << endl;
  }

  // setting the bandwidth (in MHz) of the file
  double bw_file = get_info()->get_nchan() * rate.get() * 1.0e-6;
  get_info()->set_bandwidth(bw_file);

  // getting the RA and DEC of the beam
  Attribute<double> radeg = beam.pointRA();
  if (verbose) {
    if (radeg.exists())
      cerr << "RA=" << setprecision(20) << radeg.get() << " deg" << endl;
    else cerr << "RA undefined" << endl;
  }

  Attribute<double> decdeg = beam.pointDEC();
  if (verbose) {
    if (decdeg.exists())
      cerr << "DEC=" << setprecision(20) << decdeg.get() << " deg" << endl;
    else cerr << "DEC undefined" << endl;
  }

  // setting position of the pulsar
  if (radeg.exists() && decdeg.exists()) {
   sky_coord position;
   position.setRadians((radeg.get()/180.)*M_PI, (decdeg.get()/180.)*M_PI);
   get_info()->set_coordinates(position);
  }

  // Checks
  if (verbose) {
    cerr << "*****************************************************" << endl;
    cerr << "Telescope = " << get_info()->get_telescope() << endl;
    cerr << "Backend = " << get_info()->get_machine() << endl;
    cerr << "Target = " << get_info()->get_source() << endl;
    cerr << "Number of CHANNELS = " << get_info()->get_nchan() << endl;
    cerr << "SWAP = " << get_info()->get_swap() << endl;
    cerr << "NSUB_SWAP = " << get_info()->get_nsub_swap() << endl;
    cerr << "CENTER FREQ = " << setprecision(20) << get_info()->get_centre_frequency() << endl;
    cerr << "Center freq of first channel = " << setprecision(20) << get_info()->get_centre_frequency(0) << endl;
    cerr << "Center freq of last channel  = " << setprecision(20) << get_info()->get_centre_frequency(get_info()->get_nchan()-1) << endl;
    cerr << "BANDWIDTH = " << setprecision(20) << get_info()->get_bandwidth() << endl;
    unsigned unswap_chan = get_info()->get_unswapped_ichan(0);
    cerr << "0 unswapped ichan = " << unswap_chan << endl;
    cerr << "BASE FREQUENCY = " << setprecision(20) << get_info()->get_base_frequency() << endl;
    cerr << "COORDINATES = " << get_info()->get_coordinates() << endl;
    cerr << "*****************************************************" << endl;
  }

  if (verbose) {
    if (coord.exists())
    {
      Coordinate* c = coord.coordinate( spectral_dim );
      NumericalCoordinate* num = dynamic_cast<NumericalCoordinate*> (c);

      if (num)
      {
        Attribute< std::vector<double> > world = num->axisValuesWorld();
        if (world.exists())
        {
          cerr << "SANITY CHECK" << endl;
          std::vector<double> w = world.get();
          cerr << "Size of the freq array = " << w.size() << endl;
          cerr << "-----------------------------------------------------" << endl;
          for (unsigned i=0; i<w.size(); i++)
            if (w[i] != 1000000.0 * get_info()->get_centre_frequency(i))
              cerr << "NOT EQUAL: " << setprecision(20) << w[i] << " != " << setprecision(20) << 1000000.0 * get_info()->get_centre_frequency(i) << endl;
          cerr << "*****************************************************" << endl;
        }
      }
    }
  }

  // OPEN ALL FILES

  handle = new Handle;
  handle->bf_file.resize( stokes_npol );
  handle->bf_stokes.resize( stokes_npol );

  // find which file in set was passed to this open function
  string fname (filename);
  size_t found = fname.rfind("_S");
  if (stokes_npol > 1 && found == string::npos)
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file",
		 "non-conforming filename does not contain the string \"_S\"");
  
  unsigned istokes = fname[ found+2 ] - '0';

  if (verbose) cerr << "Stokes = " << istokes << endl;

  for (unsigned i=0; i<stokes_npol; i++)
  {
    if (i == istokes)
      {
        handle->bf_file[i] = bf_file;
        handle->bf_stokes[i] = stokes;
      }
    else
      {
        fname[ found+2 ] = '0' + i;
        if (verbose) cerr << "opening " << fname << endl;
        BF_File* the_file = new BF_File (fname);
        for (sap_index=0; sap_index<nsap.get(); sap_index++) {
                if (the_file->subArrayPointing(sap_index).exists()) break;
        }
        BF_SubArrayPointing sap = the_file->subArrayPointing(sap_index);
        for (tab_index=0; tab_index<nbeam.get(); tab_index++) {
                if (sap.beam(tab_index).exists()) break;
        }
        BF_BeamGroup beam = sap.beam(tab_index);

        BF_StokesDataset* the_stokes = new BF_StokesDataset (beam.stokes(i));

        handle->bf_file[i] = the_file;
        handle->bf_stokes[i] = the_stokes;
      }
  }

}

void dsp::LOFAR_DALFile::close ()
{
  // delete everything
  handle = 0;
}

void dsp::LOFAR_DALFile::rewind ()
{
  end_of_data = false;
  current_sample = 0;

  seek (0,SEEK_SET);

  last_load_ndat = 0;
}



//! Load bytes from shared memory
int64_t dsp::LOFAR_DALFile::load_bytes (unsigned char* buffer, uint64_t bytes)
{
  if (verbose)
    cerr << "LOFAR_DALFile::load_bytes " << bytes << " bytes" << endl;

  unsigned nstokes = handle->bf_file.size();

  uint64_t nfloat = bytes / sizeof(float);
  uint64_t nsamp = nfloat / (get_info()->get_nchan() * nstokes);

  vector<size_t> pos (2);
  pos[0] = current_sample;
  pos[1] = 0;
  
  for (unsigned istokes=0; istokes < nstokes; istokes++)
  {
    // cerr << "load_bytes " << istokes << endl;
    float* outbuf = reinterpret_cast<float*> (buffer);
    handle->bf_stokes[istokes]->get2D (pos, outbuf, nsamp, get_info()->get_nchan());
    buffer += nsamp * get_info()->get_nchan() * sizeof(float);
  }
  
  return bytes;
}

//! Adjust the shared memory pointer
int64_t dsp::LOFAR_DALFile::seek_bytes (uint64_t bytes)
{
  return bytes;
}
