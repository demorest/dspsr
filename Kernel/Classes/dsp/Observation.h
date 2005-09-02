//-*-C++-*-
#ifndef __Observation_h
#define __Observation_h

#include <string>
#include <vector>
#include <typeinfo>

#include <stdio.h>

#include "Header.h"
#include "Error.h"
#include "Reference.h"
#include "sky_coord.h"
#include "Types.h"
#include "MJD.h"
#include "environ.h"

#include "dsp/dspExtension.h"
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

    //! Same as operator= but takes a pointer
    virtual void copy(const Observation* obs)
    { operator=( *obs ); }

    //! Swaps the 2 Observations.  Returns '*this'
    Observation& swap_data(Observation& obs);

    //! Print the Observation out
    virtual void print(){ cout << obs2string() << endl; }
    
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

    //! digitizer thresholds 
    virtual void set_thresh () { thresh = new float [1] ; thresh[0] = 0.0;}
    //! get a pointer to the thresholds: mean and sigma for each channel
    float * get_thresh () {return thresh;}
    
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

    //! Set true if centre channel is centred on centre frequency
    //! i.e. to get centre frequency of first channel:
    //!double dsp::Observation::get_base_frequency (){
    //!  if (dc_centred) return centre_frequency - 0.5*bandwidth;
    //!  else return centre_frequency - 0.5*bandwidth + 0.5*bandwidth/double(nchan);
    //!}
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

    //! Set the cal frequency
    void set_calfreq (double _calfreq) {calfreq = _calfreq;}

    //! get the calfreq
    double get_calfreq() const {return calfreq;} 

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
    //! The supplied dsp::Observation is for pinching the 'm'/'n' part of the 'identifier' attribute
    static string get_default_id (const MJD& mjd, const Observation* obs=0);

    //! Returns default_id (start_time);
    string get_default_id () const;

    //! Returns a string describing the state of the data
    string get_state_as_string () const;

    //! Returns the centre frequency of the first channel in MHz
    double get_base_frequency () const;

    //! Returns the minimum and maximum centre frequencies in MHz
    void get_minmax_frequencies (double& min, double& max) const;

    //! Change the start time by the number of time samples specified
    void change_start_time (int64 _ndat);

    //! Convenience function for returning the duration in seconds of the Observation
    double get_duration() const { return (get_end_time()-get_start_time()).in_seconds(); }

    //! Returns the number of samples 'latter' follows 'this' by.  (Positive means 'latter' starts later.)
    int64 samps_diff(const Observation* latter) const
    { return int64((latter->get_start_time() - get_start_time()).in_seconds() * get_rate()); }

    //! Return the end time of the trailing edge of the last time sample
    // Returns correct answer if ndat=rate=0 and avoids division by zero
    virtual MJD get_end_time () const
    { if( ndat==0 ) return start_time; return start_time + double(ndat) / rate; }

    //! Multiply scale by factor
    void rescale (double factor) { scale *= factor; }

    //! Return the size in bytes of nsamples time samples
    uint64 get_nbytes (uint64 nsamples) const
      { return (nsamples*get_nbit()*get_npol()*get_nchan()*get_ndim())/8; }

    //! Return the size in bytes of ndat time samples
    uint64 get_nbytes () const
      { return get_nbytes (get_ndat()); }

    uint64 verbose_nbytes (uint64 nsamples) const;
    
    //! Return the size in bytes of one time sample
    float get_nbyte () const
      { return float(nbit*get_npol()*get_nchan()*get_ndim()) / 8.0; }

    //! Return the number of samples in nbytes bytes
    uint64 get_nsamples (uint64 nbytes) const
      { return (nbytes * 8)/(nbit*get_npol()*get_nchan()*get_ndim()); }

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

    //! Returns all information contained in this class into the return value
    string obs2string() const;
    
    //! Converts the class information into a Header
    //! If 'hdr' is non-null, that Header is written to but its size, id and version aren't set
    Reference::To<Header> obs2Header(Header* hdr=0) const;

    //! Writes all information contained in this class into the specified filename
    void obs2file(string filename, int64 offset) const;
    
    //! Writes all information contained in this class into the file descriptor
    void obs2file(int fd, int64 offset,int whence=SEEK_SET) const;

    //! Opposite of obs2file
    void file2obs(string filename, int64 offset=0);

    //! Opposite of obs2file
    void file2obs(int fd, int64 offset, int whence=SEEK_SET);

    //! Initializes the Observation from a parsed-Header
    void Header2obs(Reference::To<Header> hdr);

    //! Set all attributes to null default
    void init ();

    //! Returns the version number to put in the Header when writing out
    float get_version() const { return 2.0; }

    //! Returns a pointer to the dspExtension
    //! If the dspExtension is not stored this throws an Error
    template<class ExtensionType>
    ExtensionType* get();

    //! Returns a pointer to the dspExtension
    //! If the dspExtension is not stored this throws an Error
    template<class ExtensionType>
    const ExtensionType* get() const;

    //! Returns true if the given dspExtension is stored
    //! Call like: if( obs->has<MiniExtension>() )...
    template<class ExtensionType>
    bool has() const;

    //! Returns true if one of the stored dspExtensions has this name
    bool has(string extension_name);

    //! Adds a dspExtension
    void add(dspExtension* extension);

    //! Removes a dspExtension
    Reference::To<dspExtension> remove_extension(string ext_name);

    //! Returns the number of dspExtensions currently stored
    unsigned get_nextensions() const;

    //! Returns the i'th dspExtension stored
    dspExtension* get_extension(unsigned iext);

    //! Returns the i'th dspExtension stored
    const dspExtension* get_extension(unsigned iext) const;

  protected:

    /* PLEASE: if you add more attributes to the dsp::Observation class then please modify obs2Header(), obs2string(), obs2file(), file2obs() appropriately!  */
    /* HSK 4/12/04 dspExtensions not currently written out */
    //! Extra stuff
    vector<Reference::To<dspExtension> > extensions;
    
    //! Tempo telescope code
    char telescope;

    //! Source name.  If a pulsar, should be J2000
    string source;

    //! Centre frequency of band-limited signal in MHz
    double centre_frequency;

    //! Bandwidth of signal in MHz (-ve = lsb; +ve = usb)
    double bandwidth;

    //! Pointer to the digitizer thresholds
    float * thresh;

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

    /* PLEASE: if you add more attributes to the dsp::Observation class then please modify obs2Header(), obs2string(), obs2file(), file2obs() appropriately!  */

    //! Worker function for file2obs that seeks through to correct place in file
    virtual void file_seek(int fd, int64 offset,int whence);
  
    //! Determines the version of the dsp::Observation printout
    float get_version(int fd);

    //! Old pre-Header version of file2obs()
    void old_file2obs(int fd);

  private:
    /////////////////////////////////////////////////////////////////////
    // Private variables should only be accessed by set/get at all times!

    //! Number of time samples in container
    //! This is private so that classes that inherit from Observation that have nbit%8 != 0
    //! can enforce resizes/set_ndat's so that ndat*ndim is always an integer number of bytes
    uint64 ndat;

    //! Dimension of each datum
    //! This is private so that classes that inherit from Observation that have nbit%8 != 0
    //! can enforce set_ndim's so that ndat*ndim is always an integer number of bytes
    unsigned ndim;

    //! Number of frequency channels across bandwidth
    unsigned nchan;

    //! Number of polarizations
    unsigned npol;

    //! Number of bits per value
    unsigned nbit;

    //! The calfrequency
    double calfreq;

  };

  class ObservationPtr {
  public:
    Observation* ptr;

    Observation& operator * () const
    { if(!ptr) throw Error(InvalidState,"dsp::ObservationPtr::operator*()","You have called operator*() when ptr is NULL"); return *ptr; }
    Observation* operator -> () const 
    { if(!ptr) throw Error(InvalidState,"dsp::ObservationPtr::operator*()","You have called operator*() when ptr is NULL"); return ptr; }
        
    ObservationPtr& operator=(const ObservationPtr& obs){ ptr = obs.ptr; return *this; }

    ObservationPtr(const ObservationPtr& obs){ ptr = obs.ptr; }
    ObservationPtr(Observation* _ptr){ ptr = _ptr; }
    ObservationPtr(){ ptr = 0; }

    bool operator < (const ObservationPtr& obs) const
    { return ptr->get_centre_frequency() < obs->get_centre_frequency(); }

    ~ObservationPtr(){ }
  };

}

//! Returns a pointer to the dspExtension
//! If the dspExtension is not stored this throws an Error
template<class ExtensionType>
ExtensionType* dsp::Observation::get(){
  ExtensionType* ret = 0;

  for( unsigned i=0; i<extensions.size(); i++){
    ret = dynamic_cast<ExtensionType*>(extensions[i].get());
    if( ret )
      return ret;
  }

  throw Error(InvalidState,"dsp::Observation::get()",
	      "Could not find a matching extension of the %d stored for '%s'",
	      extensions.size(), typeid(ret).name());
  
  return 0;
}

//! Returns a pointer to the dspExtension
//! If the dspExtension is not stored this throws an Error
template<class ExtensionType>
const ExtensionType* dsp::Observation::get() const{
  ExtensionType* ret = 0;

  for( unsigned i=0; i<extensions.size(); i++){
    ret = dynamic_cast<ExtensionType*>(extensions[i].get());
    if( ret )
      return ret;
  }

  throw Error(InvalidState,"dsp::Observation::get()",
	      "Could not find a matching extension of the %d stored for '%s'",
	      extensions.size(), typeid(ret).name());
  
  return 0;
}

//! Returns true if the given dspExtension is stored
//! Call like: if( obs->has<MiniExtension>() )...
template<class ExtensionType>
bool dsp::Observation::has() const{
  for( unsigned i=0; i<extensions.size(); i++)
    if( dynamic_cast<ExtensionType*>(extensions[i].get()) )
      return true;

  return false;
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
