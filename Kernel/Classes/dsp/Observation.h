//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Observation.h,v $
   $Revision: 1.59 $
   $Date: 2003/09/01 07:17:17 $
   $Author: hknight $ */

#ifndef __Observation_h
#define __Observation_h

#include <string>

#include <stdio.h>

#include "Reference.h"
#include "sky_coord.h"
#include "Types.h"
#include "MJD.h"
#include "dsp/dsp.h"

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

    //! Swaps the 2 Observations.  Returns '*this'
    Observation& swap_data(Observation& obs);
    
    //! Set the type of receiver feeds
    void set_basis (Signal::Basis _basis) { basis = _basis; }
    //! Return the type of receiver feeds
    Signal::Basis get_basis () const { return basis; }

    //! Set the state of the signal
    virtual void set_state (Signal::State _state);
    //! Return the state of the signal
    Signal::State get_state () const { return state; }

    //! Set the source type
    void set_type (Signal::Source _type) { type = _type; }
    //! Return the source type
    Signal::Source get_type () const { return type; }

    //! Set the dimension of each datum
    virtual void set_ndim (unsigned _ndim) { ndim = _ndim; }
     //! Return the dimension of each datum
    unsigned get_ndim () const { return ndim; }

     //! Set the number of channels into which the band is divided
    virtual void set_nchan (unsigned _nchan) { nchan = _nchan; }
    //! Return the number of channels into which the band is divided
    unsigned get_nchan () const { return nchan; }

    //! Set the number of polarizations
    virtual void set_npol (unsigned _npol) { npol = _npol; }
    //! Return the number of polarizations
    unsigned get_npol () const { return npol; }

    //! Set the number of bits per value
    virtual void set_nbit (unsigned _nbit) { nbit = _nbit; }
    //! Return the number of bits per value
    unsigned get_nbit () const { return nbit; }

    //! Set the number of time samples in container
    /*! Note that one time sample may be complex and/or vector in
      nature.  For instance, the in-phase and quadrature components of
      two orthogonal polarizations, though represented by four
      independent numbers, still represent one time sample. */
    virtual void set_ndat (uint64 _ndat) { ndat = _ndat; }
    //! Return the number of time samples in container
    uint64 get_ndat () const { return ndat; }

    //! Set the tempo telescope code
    void set_telescope_code (char telescope);
    //! Return the tempo telescope code
    char get_telescope_code () const { return telescope; }

    //! Set the source name
    void set_source (string _source) { source = _source; }
    //! Return the source name
    string get_source () const { return source; }

    //! Set the coordinates of the source
    void set_coordinates (sky_coord _coordinates) { coordinates=_coordinates; }
    //! Return the coordinates of the source
    sky_coord get_coordinates () const { return coordinates; }

    //! Set the centre frequency of the band-limited signal in MHz
    void set_centre_frequency (double cf) { centre_frequency = cf; }
    //! Return the centre frequency of the band-limited signal in MHz
    double get_centre_frequency () const{ return centre_frequency; }

    //! Returns the centre frequency of the specified channel in MHz
    double get_centre_frequency (unsigned ichan) const;

    //! Returns the centre frequency of the ichan'th frequency ordered channel in MHz.
    double get_ordered_cfreq(unsigned ichan);

    //! Return the centre frequency of the highest frequency channel below the maximum stated in MHz
    double get_highest_frequency(double max_freq=1.0e9, unsigned chanstart=0,unsigned chanend=99999);

    //! Return the centre frequency of the lowest frequency channel in MHz
    double get_lowest_frequency(double min_freq=0.0, unsigned chanstart=0,unsigned chanend=99999);

    //! Set the bandwidth of signal in MHz (-ve = lsb; +ve = usb)  lsb means that the highest RF frequency is in channel 0
    void set_bandwidth (double _bandwidth) { bandwidth = _bandwidth; }
    //! Return the bandwidth of signal in MHz (-ve = lsb; +ve = usb)  lsb means that the highest RF frequency is in channel 0
    double get_bandwidth () const { return bandwidth; }

    //! Set the start time of the leading edge of the first time sample
    void set_start_time (MJD _start_time) { start_time = _start_time; }
    //! Return the start time of the leading edge of the first time sample
    MJD get_start_time () const { return start_time; }
    
    //! Set the sampling rate (time samples per second in Hz)
    void set_rate (double _rate) { rate = _rate; }
    //! Return the sampling rate (time samples per second in Hz)
    double get_rate () const { return rate; }

    //! Set the amount by which data has been scaled
    void set_scale (double _scale) { scale = _scale; }
    //! Return the amount by which data has been scaled
    double get_scale () const { return scale; }

    //! Set true if frequency channels are out of order (band swappped)
    void set_swap (bool _swap) { swap = _swap; }
    //! Return true if frequency channels are out of order (band swappped)
    bool get_swap () const { return swap; }

    //! Set true if centre channel is centred on centre frequency (doesn't actually sufficiently describe band so will be deprecated soon)
    void set_dc_centred (bool _dc_centred) { dc_centred = _dc_centred; }
    bool get_dc_centred () const { return dc_centred; }

    //! Set the observation identifier
    void set_identifier (string _identifier) { identifier = _identifier; }
    //! Return the observation identifier
    string get_identifier () const { return identifier; }

    //! Set the instrument used to record signal
    void set_machine (string _machine) { machine = _machine; }
    //! Return the instrument used to record signal
    string get_machine () const { return machine; }

    //! Returns the DM to which the data has been dedispersed (in-channel dispersion)
    double get_dispersion_measure () const { return dispersion_measure; }

    //! Set the record of what DM the data is dedispersed (in-channel dispersion)
    void set_dispersion_measure (double dm) { dispersion_measure = dm; }

    //! Add on extra DM units to the dispersion measure (in-channel smearing)
    void change_dispersion_measure(double dm) { dispersion_measure += dm; }

    //! Returns the DM to which the data has been dedispersed (between-channel dispersion)
    double get_between_channel_dm () const { return between_channel_dm; }

    //! Set the record of what DM the data is dedispersed (between-channel dispersion)
    void set_between_channel_dm (double dm) { between_channel_dm = dm; }

    //! Add on extra DM units to the dispersion measure (between-channel smearing)
    void change_between_channel_dm(double dm) { between_channel_dm += dm; }

    //! Set the observation mode
    void set_mode (string _mode) { mode = _mode; }
    //! Return the observation mode
    string get_mode () const { return mode; }

    //! Whether data is in 'Time' or 'Fourier' or some variant that starts with 'Fourier'.  Classes that change this are PowerSpectrumMKL, PowerSpectrumFFTW, PowerTwoFFTW, PowerTwoMKL.  BasicPlotter and/or Plotter uses it too I think.  HSK 21/11/02
    //! Also, H_BandPass sets this as bandpass
    string get_domain() const { return domain; }

    //! Whether data is in 'Time' or 'Fourier' domain 
    void set_domain(string _domain){ domain = _domain; }

    //! Inquire the last format on disk the dat was stored on
    string get_last_ondisk_format() const { return last_ondisk_format; }

    //! Set the last format on disk the dat was stored on
    void set_last_ondisk_format(string _last_ondisk_format){ last_ondisk_format = _last_ondisk_format; }

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

    //! Returns the centre frequency of the first channel in MHz
    double get_base_frequency () const;

    //! Returns the minimum and maximum centre frequencies in MHz
    void get_minmax_frequencies (double& min, double& max) const;

    //! Change the start time by the number of time samples specified
    void change_start_time (int64 ndat);

    //! Returns the number of samples 'latter' follows 'this' by.  (Positive means 'latter' starts later.)
    int64 samps_diff(const Observation* latter) const
    { return int64((latter->get_start_time() - get_start_time()).in_seconds() * get_rate()); }

    //! Return the end time of the trailing edge of the last time sample
    virtual MJD get_end_time () const
    { return start_time + double(ndat) / rate; }

    //! Multiply scale by factor
    void rescale (double factor) { scale *= factor; }

    //! Return the size in bytes of nsamples time samples
    uint64 get_nbytes (uint64 nsamples) const
      { return (nsamples*nbit*npol*nchan*get_ndim())/8; }

    //! Return the size in bytes of ndat time samples
    uint64 get_nbytes () const
      { return get_nbytes (ndat); }

    uint64 verbose_nbytes (uint64 nsamples) const;
    
    //! Return the size in bytes of one time sample
    float get_nbyte () const
      { return float(nbit*npol*nchan*get_ndim()) / 8.0; }

    //! Return the number of samples in nbytes bytes
    uint64 get_nsamples (uint64 nbytes) const
      { return (nbytes * 8)/(nbit*npol*nchan*get_ndim()); }

    //! Constructs the CPSR2-header parameter, OFFSET
    uint64 get_offset();

    //! Returns true if the signal may be integrated
    /* This returns a flag that is true if the Observations may be combined 
       It doesn't check the start times- you have to do that yourself!
    */
    //! If ichan>=0 && ipol>=0 it means 'obs' should only be compared
    //! with that particular ichan/ipol 
    bool combinable (const Observation& obs, bool different_bands=false,
		     bool combinable_verbose=false,
		     int ichan=-1,int ipol=-1) const;

    //! The same as combinable, but use this for two bands of differing sky frequencies
    bool bands_combinable(const Observation& obs,bool combinable_verbose=false) const;

    //! Called by combinable() to see if bands are adjoining if the 'different_bands' variable is set to true
    bool bands_adjoining(const Observation& obs) const;

    //! Called by combinable(), and does every check but the bands_adjoining() check
    bool ordinary_checks(const Observation& obs, bool different_bands=false,
			 bool combinable_verbose=false,
			 int ichan=-1,int ipol=-1) const;

    //! Return true if test_rate is withing 1% of the rate attribute
    virtual bool combinable_rate (double test_rate) const;

    //! Return true if the first sample of next follows the last sample of this
    //! If ichan>=0 && ipol>=0 calls combinable() for only that chanpol
    bool contiguous (const Observation& next, bool verbose_on_failure=true,
		     int ichan=-1,int ipol=-1) const;

    //! Sets the feed type based on the telescope and centre frequency
    void set_default_basis ();

    //! Returns all information contained in this class into the string info_string
    bool obs2string(string& info_string) const;
    
    //! Writes of all information contained in this class into the fptr at the current file offset.  Does no seeking etc.
    bool obs2file(FILE* fptr);

    //! Opposite of obs2file
    bool file2obs(FILE* fptr);

    //! Set all attributes to null default
    virtual void init ();

  protected:

    /* PLEASE: if you add more attributes to the dsp::Observation class then please modify obs2string(), file2obs() and string2obs() appropriately! */
    
    //! Number of time samples in container
    uint64 ndat;

    //! Tempo telescope code
    char telescope;

    //! Source name.  If a pulsar, should be J2000
    string source;

    //! Centre frequency of band-limited signal in MHz
    double centre_frequency;

    //! Bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    double bandwidth;

    //! Number of frequency channels across bandwidth
    unsigned nchan;

    //! Number of polarizations
    unsigned npol;

    //! Dimension of each datum
    unsigned ndim;

    //! Number of bits per value
    unsigned nbit;

    //! Type of signal source (Linear or Circular)
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

    //! The DM TimeSeries has been dedispersed to (in-channel smearing) (latest ChannelSum only)
    double dispersion_measure;

    //! The DM the channels have been shifted by
    double between_channel_dm;

    //! Whether data is in 'Time' or 'Fourier' or some variant that starts with 'Fourier'.  Classes that change this are PowerSpectrumMKL, PowerSpectrumFFTW, PowerTwoFFTW, PowerTwoMKL.  BasicPlotter and/or Plotter uses it too I think.  HSK 21/11/02
    string domain;

    //! The last format the data was stored as ("raw","CoherentFB","Digi" etc)
    string last_ondisk_format;

    /* PLEASE: if you add more attributes to the dsp::Observation class then please modify obs2string() and file2obs() appropriately! */

  };

}

#ifdef ACTIVATE_MPI

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
