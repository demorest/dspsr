/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Observation.h,v $
   $Revision: 1.1 $
   $Date: 2002/06/30 05:09:01 $
   $Author: pulsar $ */

#ifndef __Observation_h
#define __Observation_h

#include <string>

#include "MJD.h"
#include "sky_coord.h"

#ifdef MPI
#include "mpi.h"
#endif

#define TELID_PKS  '7'
#define TELID_ATCA '2'
#define TELID_TID  '6'
#define TELID_ARECIBO '3'
#define TELID_HOBART '4'

namespace dsp {

  //! Base class for containers of band-limited time-varying signals
  class Observation {

  public:

    //! Possible states of the data
    enum State { Unknown,
		 //! Nyquist sampled voltages (real)
		 Nyquist,
		 //! In-phase and Quadrature sampled voltages (complex)
		 Analytic,
		 //! Square-law detected power
		 Detected,
		 //! PP, QQ, Re[PQ], Im[PQ]
		 Coherence,
		 //! Stokes I,Q,U,V
		 Stokes
    };

    //! Receiver feed types
    enum Feed { Invalid = -1,
		Circular = 0,
		Linear = 1 };

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

    //! Set the number of time samples in container
    /*! Note that one time sample may be complex and/or vector in
      nature.  For instance, the in-phase and quadrature components of
      two orthogonal polarizations, though represented by four
      independent numbers, still represent one time sample. */
    void set_ndat (int64 _ndat) { ndat = _ndat; }
    //! Return the number of time samples in container
    int64 get_ndat () const { return ndat; }

    //! Set the centre frequency of the band-limited signal in MHz
    void set_centre_frequency (double cf) { centre_frequency = cf; }
    //! Return the centre frequency of the band-limited signal in MHz
    double get_centre_frequency () const { return centre_frequency; }

    //! Returns the centre frequency of the specified channel
    double get_centre_frequency (int ichan) const;


    //! Set the bandwidth of signal in MHz
    void set_bandwidth (double _bandwidth) { bandwidth = _bandwidth; }
    //! Return the bandwidth of signal in MHz
    double get_bandwidth () const { return bandwidth; }

    //! Set the number of channels into which the band is divided
    void set_nchan (int _nchan) { nchan = _nchan; }
    //! Return the number of channels into which the band is divided
    int get_nchan () const { return nchan; }

    //! Set the number of polarizations
    void set_npol (int _npol) { npol = _npol; }
    //! Return the number of polarizations
    int get_npol () const { return npol; }

    //! Returns the dimension of the data (complex or real)
    int get_ndim () const { if (state==Analytic) return 2; else return 1; }

    //! Set the type of receiver feeds
    void set_feedtype (Feed _feedtype) { feedtype = _feedtype; }
   //! Return the type of receiver feeds
    Feed get_feedtype () const { return feedtype; }

    //! Set the start time of the leading edge of the first time sample
    void set_start_time (MJD _start_time) { start_time = _start_time; }
    //! Return the start time of the leading edge of the first time sample
    MJD get_start_time () const { return start_time; }
    
    //! Change the start time by the number of time samples specified
    void change_start_time (int64 ndat);

    //! Return the end time of the trailing edge of the last time sample
    MJD get_end_time () const { return start_time + double(ndat + 1) / rate; }

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

    //! Set the state of the signal
    void set_state (State _state) { state = _state; }
    //! Return the state of the signal
    State get_state () const { return state; }

    //! Change the state and correct other attributes accordingly
    virtual void change_state (State new_state);

    //! Returns true if state is Detected, Coherence, or Stokes
    bool get_detected () const { return state >= Detected; }

    //! Set true if frequency channels are out of order (band swappped)
    void set_swap (bool _swap) { swap = _swap; }
    //! Return true if frequency channels are out of order (band swappped)
    bool get_swap () const { return swap; }

    //! Set true if centre channel is centred on centre frequency
    void set_dc_centred (bool _dc_centred) { dc_centred = _dc_centred; }
    //! Return true if centre channel is centred on centre frequency
    bool get_dc_centred () const { return dc_centred; }

    //! Set the tempo telescope code
    void set_telescope (char _telescope) { telescope = _telescope; }
    //! Return the tempo telescope code
    char get_telescope () const { return telescope; }

    //! Set the source name
    void set_source (string _source) { source = _source; }
    //! Return the source name
    string get_source () const { return source; }

    //! Set the observation identifier
    void set_identifier (string _identifier) { identifier = _identifier; }
    //! Return the observation identifier
    string get_identifier () const { return identifier; }

    //! Set the observation mode
    void set_mode (string _mode) { mode = _mode; }
    //! Return the observation mode
    string get_mode () const { return mode; }

    //! Set the coordinates of the source
    void set_position (sky_coord _position) { position = _position; }
    //! Return the coordinates of the source
    sky_coord get_position () const { return position; }

    //! Set the instrument used to record signal
    void set_machine (string _machine) { machine = _machine; }
    //! Return the instrument used to record signal
    string get_machine () const { return machine; }

    //! Returns a convenient id string for a given MJD
    static string get_default_id (const MJD& mjd);

    //! Returns default_id (start_time);
    string get_default_id () const;

    //! Returns a string describing the state of the data
    string get_state_str () const;

    //! Returns true if the signal may be integrated
    bool combinable (const Observation& obs);

    //! Sets the feed type based on the telescope and centre frequency
    void set_default_feedtype ();

  protected:

    //! Number of time samples in container
    int64 ndat;

    //! Centre frequency of band-limited signal in MHz
    double centre_frequency;

    //! Bandwidth of signal in MHz
    double bandwidth;

    //! Number of frequency channels across bandwidth
    int nchan;

    //! Number of polarizations
    int npol;

    //! Type of receiver feeds
    Feed feedtype;

    //! Start time of the leading edge of the first time sample
    MJD start_time;
    
    //! Time samples per second in Hz
    double rate;

    //! Amount by which data has been scaled
    double scale;

    //! State of the signal
    State state;

    //! Flag set when frequency channels are out of order (band swappped)
    bool swap;

    //! Flag set when centre channel is centred on centre frequency
    bool dc_centred;

    //! Tempo telescope code
    char telescope;

    //! Source name.  If a pulsar, should be J2000
    string source;

    //! Observation identifier
    string identifier;

    //! Observation mode
    string mode;

    //! Coordinates of the source
    sky_coord position;

    //! Instrument used to record signal
    string machine;

    //! Set all attributes to null default
    void init ();
  };

}

#endif // ! __Observation_h
