/***************************************************************************
 *
 *   Copyright (C) 2005-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

using namespace std;

#include "dsp/LOFAR_DALFile.h"

#include "lofar/BF_File.h"
using namespace DAL;

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
  DAL::BF_File* bf_file = new DAL::BF_File (filename);

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

  DAL::BF_File* bf_file = new DAL::BF_File (filename);

  Attribute<std::string> telescope = bf_file->telescope();

  if (telescope.exists())
    cerr << "LOFAR_DALFile::open_file telescope=" << telescope.get() << endl;

  Attribute<std::string> observer = bf_file->observer();

  if (observer.exists())
    cerr << "LOFAR_DALFile::open_file OBSERVER=" << observer.get() << endl;

  Attribute<std::string> target = bf_file->target();

  if (target.exists())
    cerr << "LOFAR_DALFile::open_file TARGET=" << target.get() << endl;
  else
    cerr << "TARGET does not exist" << endl;

  Attribute<double> freq = bf_file->observationFrequencyCenter();

  if (freq.exists())
    cerr << "LOFAR_DALFile::open_file observation frequency=" << freq.get() << endl;



  Attribute<unsigned> nsap = bf_file->nofSubArrayPointings();
  if (nsap.exists())
    cerr << "LOFAR_DALFile::open_file number of SAPs=" << nsap.get() << endl;
  else
    cerr << "observation nsaps does not exist" << endl;


  cerr << endl << "*****************" << endl << endl;


  BF_SubArrayPointing sap = bf_file->subArrayPointing (0);

  Attribute<double> bw2 = sap.subbandWidth();

  if (bw2.exists())
    cerr << "LOFAR_DALFile::open_file sap subbandwidth=" << bw2.get() << endl;

  Attribute<unsigned> nbeam = sap.nofBeams();
  if (nbeam.exists())
    cerr << "LOFAR_DALFile::open_file number of beams=" << nbeam.get() << endl;
  else
    cerr << "sap nbeams does not exist" << endl;

  Attribute<unsigned> nchan = sap.channelsPerSubband();
  if (nchan.exists())
    cerr << "LOFAR_DALFile::open_file number of channels=" << nchan.get() << endl;
  else
    cerr << "sap channelsPerSubband does not exist" << endl;


  cerr << endl << "*****************" << endl << endl;


  BF_BeamGroup beam = sap.beam (0);

  Attribute<double> freq2 = beam.beamFrequencyCenter();

  if (freq2.exists())
    cerr << "LOFAR_DALFile::open_file beam frequency=" << freq2.get() << endl;

  Attribute< std::vector<std::string> > targets = beam.target();
  if (targets.exists())
    {
      std::vector<std::string> t = targets.get();
      cerr << "targets size=" << t.size() << endl;
    }
  else
    cerr << "beam target does not exist" << endl;

  cerr << endl << "*****************" << endl << endl;


  BF_StokesDataset* stokes = 0;

  for (unsigned i=0; i<4; i++)
  {
    BF_StokesDataset tmp = beam.stokes(i);
    if (tmp.exists())
      stokes = new BF_StokesDataset (beam.stokes(i));
  }
      
  Attribute<std::string> stokesC = stokes->stokesComponent();
  if (stokesC.exists())
    cerr << "stokes component=" << stokesC.get() << endl;

  // Attribute< std::vector<unsigned> >    nofChannels();

  Attribute<unsigned> nsub = stokes->nofSubbands();
  if (nsub.exists())
    cerr << "nsub=" << nsub.get() << endl;
  else
    cerr << "stokes nofSubbands not defined" << endl;

  Attribute< std::vector<unsigned> > nofchan = stokes->nofChannels();
  if (nchan.exists())
  {
    std::vector<unsigned> nchan = nofchan.get();
    cerr << "stokes nofChannels size=" << nchan.size() << endl;
    for (unsigned i=0; i<nchan.size(); i++)
      cerr << "stokes nofChannels[" << i << "]=" << nchan[i] << endl;
  }
  else
    cerr << "stokes nofChannels not defined" << endl;


  size_t ndim= stokes->ndims();

  cerr << "stokes ndim=" << ndim << endl;

  std::vector<std::string> files = stokes->externalFiles();
  for (unsigned i=0; i<files.size(); i++)
    cerr << "files[" << i << "]=" << files[i] << endl;

  /* **********************************************************************
   *
   *
   *
   *
   * ********************************************************************** */

  // set Observation attributes

  Attribute<unsigned> nsamp = stokes->nofSamples();
  if (nsamp.exists())
    info.set_ndat( nsamp.get() );

  Attribute<bool> volts = beam.complexVoltages();
  if (volts.exists() && volts.get() == 1)
    info.set_ndim (2);
  else
    info.set_ndim (1);
  
  // check for which coordinate is Spectral

  unsigned spectral_dim = 1;

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
  info.set_nchan( dims[spectral_dim] );
  

  Attribute<unsigned> npol = beam.nofStokes();

  unsigned stokes_npol = 1;
  if (npol.exists())
    stokes_npol = npol.get();

  if (stokes_npol == 1)
  {
    info.set_npol (1);
    info.set_state( Signal::Intensity );
  }
  else
  {
    if (info.get_ndim() == 2)  // complexVoltages == true
    {
      info.set_npol (2);
      info.set_state( Signal::Analytic );
    }
    else
    {
      info.set_npol (4);
      info.set_state( Signal::Stokes );
    }
  }

  info.set_nbit (32);


  Attribute<double> cfreq = beam.beamFrequencyCenter();
  if (!cfreq.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file",
		 "beamFrequencyCenter not defined");

  // assuming cfreq is in Hz
  info.set_centre_frequency( cfreq.get() * 1e-6 );



  Attribute<double> bw = bf_file->bandwidth();

  if (!bw.exists())
    throw Error (InvalidState, "dsp::LOFAR_DALFile::open_file",
		 "bandwidth not defined");
  
  // assuming bandwidth is in MHz
  info.set_bandwidth( bw.get() );

  info.set_dc_centred( true );

  Attribute<double> mjd = bf_file->observationStartMJD();
  if (mjd.exists())
    info.set_start_time( MJD(mjd.get()) );

  cerr << "MJD=" << info.get_start_time() << endl;

  Attribute<double> cRate = sap.clockRate();
  if (cRate.exists())
    cerr << "clockRate=" << cRate.get() << endl;
  else
    cerr << "clockRate undefined" << endl;

  Attribute<double> sRate = sap.samplingRate();
  if (sRate.exists())
    cerr << "samplingRate=" << sRate.get() << endl;
  else
    cerr << "samplingRate undefined" << endl;

  Attribute<double> sTime = sap.samplingTime();
  if (sTime.exists())
    cerr << "samplingTime=" << sTime.get() << endl;
  else
    cerr << "samplingTime undefined" << endl;


  Attribute<double> rate = sap.channelWidth();
  if (rate.exists())
    info.set_rate (rate.get());

  
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
	for (unsigned i=0; i<w.size(); i++)
	  if (w[i] != info.get_centre_frequency(i)*1e6)
	    cerr << "NOT EQUAL: " << w[i] << " != " << info.get_centre_frequency(i)
		 << endl;
      }
    }
  }

  info.set_machine( "LOFAR" );

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

  cerr << "Stokes = " << istokes << endl;

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
	cerr << "opening " << fname << endl;
	BF_File* the_file = new DAL::BF_File (fname);
	BF_SubArrayPointing sap = the_file->subArrayPointing (0);
	BF_BeamGroup beam = sap.beam (0);
	
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
  uint64_t nsamp = nfloat / (info.get_nchan() * nstokes);

  vector<size_t> pos (2);
  pos[0] = current_sample;
  pos[1] = 0;
  
  for (unsigned istokes=0; istokes < nstokes; istokes++)
  {
    // cerr << "load_bytes " << istokes << endl;
    float* outbuf = reinterpret_cast<float*> (buffer);
    handle->bf_stokes[istokes]->get2D (pos, nsamp, info.get_nchan(), outbuf);
    buffer += nsamp * info.get_nchan() * sizeof(float);
  }
  
  return bytes;
}

//! Adjust the shared memory pointer
int64_t dsp::LOFAR_DALFile::seek_bytes (uint64_t bytes)
{
  return bytes;
}
