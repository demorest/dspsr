//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Observation.h,v $
   $Revision: 1.22 $
   $Date: 2002/10/16 16:23:25 $
   $Author: wvanstra $ */

#ifndef __Observation_h
#define __Observation_h

#include <string>

#include "Reference.h"
#include "sky_coord.h"
#include "Types.h"
#include "MJD.h"

/*! \mainpage 
 
  \section intro Introduction
 
  The Baseband Data Reduction Library implements a family of C++
  classes that may be used in the loading and manipulation of
  phase-coherent observational data.  The functionality, contained in
  the dsp namespace, is divided into three main classes: data
  containers and loaders, DSP algorithms, and auxilliary routines.

  The main data container is the dsp::Timeseries class.  This class
  may hold N-bit digitized data as well as the "unpacked" floating
  point representation.  The dsp::Loader class and its children are
  used to load data into the dsp::Timeseries container.

  The main DSP algorithms are implemented by dsp::Operation and its
  sub-classes.  These operate on dsp::Timeseries and can:
  <UL>
  <LI> convert digitized data to floating points (dsp::TwoBitCorrection class)
  <LI> coherently dedisperse data (dsp::Convolution class)
  <LI> fold data using polyco (dsp::Fold class)
  <LI> etc...
  </UL>

  The auxilliary routines include classes that perform operations on
  arrays of data, such as multiplying a jones matrix frequency response
  by a complex vector spectrum (e.g. the dsp::filter class).

 */

//! Contains all Baseband Data Reduction Library classes
namespace dsp {

  //! Stores information about digital, band-limited, time-varying signals
  class Observation : public Reference::Able {

  public:

    //! Verbosity flag
    static bool verbose;

    //! Null constructor
    Observation (); 

    //! Copy constructor
    Observation (const Observation &);

    //! Virtual destructor (see Effective C++ Item 14)
    virtual ~Observation(); 

    //! Assignment operator
    Observation& operator= (const Observation&);

    //! Set the dimensions of each time sample
    /*! Parameters determine the size and interpretation of each datum */
    virtual void set_sample (Signal::State state,
			     int nchan, int npol, int ndim, int nbit);

    //! Set the state of the signal
    virtual void set_state (Signal::State _state);
    //! Return the state of the signal
    Signal::State get_state () const { return state; }

    //! Set the dimension of each datum
    virtual void set_ndim (int _ndim) { ndim = _ndim; }
     //! Return the dimension of each datum
    int get_ndim () const { return ndim; }

     //! Set the number of channels into which the band is divided
    virtual void set_nchan (int _nchan) { nchan = _nchan; }
    //! Return the number of channels into which the band is divided
    int get_nchan () const { return nchan; }

    //! Set the number of polarizations
    virtual void set_npol (int _npol) { npol = _npol; }
    //! Return the number of polarizations
    int get_npol () const { return npol; }

    //! Set the number of bits per value
    void set_nbit (int _nbit) { nbit = _nbit; }
    //! Return the number of polarizations
    int get_nbit () const { return nbit; }

    //! Set the number of time samples in container
    /*! Note that one time sample may be complex and/or vector in
      nature.  For instance, the in-phase and quadrature components of
      two orthogonal polarizations, though represented by four
      independent numbers, still represent one time sample. */
    virtual void set_ndat (int64 _ndat) { ndat = _ndat; }
    //! Return the number of time samples in container
    int64 get_ndat () const { return ndat; }

    //! Set the tempo telescope code
    void set_telescope (char telescope);
    //! Return the tempo telescope code
    char get_telescope () const { return telescope; }

    //! Set the source type
    void set_type (Signal::Source _type) { type = _type; }
    //! Return the source type
    Signal::Source get_type () const { return type; }

    //! Set the source name
    void set_source (string _source) { source = _source; }
    //! Return the source name
    string get_source () const { return source; }

    //! Set the centre frequency of the band-limited signal in MHz
    void set_centre_frequency (double cf) { centre_frequency = cf; }
    //! Return the centre frequency of the band-limited signal in MHz
    double get_centre_frequency () const { return centre_frequency; }
    //! Returns the centre frequency of the specified channel in MHz
    double get_centre_frequency (int ichan) const;

    //! Set the bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    void set_bandwidth (double _bandwidth) { bandwidth = _bandwidth; }
    //! Return the bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    double get_bandwidth () const { return bandwidth; }

    //! Set the type of receiver feeds
    void set_basis (Signal::Basis _basis) { basis = _basis; }
   //! Return the type of receiver feeds
    Signal::Basis get_basis () const { return basis; }

    //! Set the start time of the leading edge of the first time sample
    void set_start_time (MJD _start_time) { start_time = _start_time; }
    //! Return the start time of the leading edge of the first time sample
    MJD get_start_time () const { return start_time; }
    
    //! Change the start time by the number of time samples specified
    void change_start_time (int64 ndat);

    //! Return the end time of the trailing edge of the last time sample
    virtual MJD get_end_time () const
    { return start_time + double(ndat + 1) / rate; }

    //! Set the sampling rate (time samples per second in Hz)
    void set_rate (double _rate) { rate = _rate; }
    //! Return the sampling rate (time samples per second in Hz)
    double get_rate () const { return rate; }

    //! Set the amount by which data has been scaled
    void set_scale (double _scale) { scale = _scale; }
    //! Return the amount by which data has been scaled
    double get_scale () const { return scale; }

    //! Multiply scale by factor
    void rescale (double factor) { scale *= factor; }

    //! Set true if frequency channels are out of order (band swappped)
    void set_swap (bool _swap) { swap = _swap; }
    //! Return true if frequency channels are out of order (band swappped)
    bool get_swap () const { return swap; }

    //! Set true if centre channel is centred on centre frequency
    void set_dc_centred (bool _dc_centred) { dc_centred = _dc_centred; }
    //! Return true if centre channel is centred on centre frequency
    bool get_dc_centred () const { return dc_centred; }

    //! Set the observation identifier
    void set_identifier (string _identifier) { identifier = _identifier; }
    //! Return the observation identifier
    string get_identifier () const { return identifier; }

    //! Set the observation mode
    void set_mode (string _mode) { mode = _mode; }
    //! Return the observation mode
    string get_mode () const { return mode; }

    //! Set the coordinates of the source
    void set_coordinates (sky_coord _coordinates) { coordinates=_coordinates; }
    //! Return the coordinates of the source
    sky_coord get_coordinates () const { return coordinates; }

    //! Set the instrument used to record signal
    void set_machine (string _machine) { machine = _machine; }
    //! Return the instrument used to record signal
    string get_machine () const { return machine; }

    //! Returns the DM to which the data has been dedispersed
    double get_dispersion_measure () const { return dispersion_measure; }

    //! Set the record of what DM the data is dedispersed
    void set_dispersion_measure (double dm) { dispersion_measure = dm; }

    //! Change the state and correct other attributes accordingly
    virtual void change_state (Signal::State new_state);

    //! Return true if the state of the Observation is valid
    bool state_is_valid (string& reason) const;

    //! Returns true if state is Detected, Coherence, or Stokes
    bool get_detected () const;

    //! Returns a convenient id string for a given MJD
    static string get_default_id (const MJD& mjd);

    //! Returns default_id (start_time);
    string get_default_id () const;

    //! Returns a string describing the state of the data
    string get_state_as_string () const;

    //! Return the size in bytes of nsamples time samples
    int64 nbytes (int64 nsamples) const
      { return (nsamples*nbit*npol*nchan*get_ndim())/8; }
    int64 verbose_nbytes (int64 nsamples) const;
    
    //! Return the size in bytes of one time sample
    float nbyte () const
      { return float(nbit*npol*nchan*get_ndim()) / 8.0; }

    //! Return the size in bytes of ndat time samples
    int64 nbytes () const
      { return nbytes (ndat); }

    //! Return the number of samples in nbytes bytes
    int64 nsamples (int64 nbytes) const
      { return (nbytes * 8)/(nbit*npol*nchan*get_ndim()); }

    //! Returns true if the signal may be integrated
    bool combinable (const Observation& obs);

    //! Sets the feed type based on the telescope and centre frequency
    void set_default_basis ();

  protected:

    //! Number of time samples in container
    int64 ndat;

    //! Tempo telescope code
    char telescope;

    //! Source name.  If a pulsar, should be J2000
    string source;

    //! Centre frequency of band-limited signal in MHz
    double centre_frequency;

    //! Bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    double bandwidth;

    //! Number of frequency channels across bandwidth
    int nchan;

    //! Number of polarizations
    int npol;

    //! Dimension of each datum
    int ndim;

    //! Number of bits per value
    int nbit;

    //! Type of signal source
    Signal::Source type;

    //! State of the signal
    Signal::State state;

    //! Type of receiver feeds
    Signal::Basis basis;

    //! Time samples per second in Hz
    double rate;

    //! Start time of the leading edge of the first time sample
    MJD start_time;
    
    //! Amount by which data has been scaled
    double scale;

    //! Flag set when frequency channels are out of order (band swappped)
    bool swap;

    //! Flag set when centre channel is centred on centre frequency
    bool dc_centred;

    //! Observation identifier
    string identifier;

    //! Observation mode
    string mode;

    //! Instrument used to record signal
    string machine;

    //! Coordinates of the source
    sky_coord coordinates;

    //! The DM Timeseries has been dedispersed to
    double dispersion_measure;

    //! Set all attributes to null default
    void init ();
  };

}

#ifdef MPI

#include <mpi.h>

//! Return the size required to mpiPack an Observation
int mpiPack_size (const dsp::Observation&, MPI_Comm comm, int* size);

//! Pack an Observation into outbuf
int mpiPack (const dsp::Observation&,
	     void* outbuf, int outcount, int* position, MPI_Comm comm);

//! Unpack an Observation from inbuf
int mpiUnpack (void* inbuf, int insize, int* position, 
	       dsp::Observation*, MPI_Comm comm);

#endif

#endif // ! __Observation_h
