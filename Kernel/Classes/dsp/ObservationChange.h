//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __dsp_ObservationChange_h
#define __dsp_ObservationChange_h

#include "dsp/Observation.h"

namespace dsp {

  //! Stores parameters that should be changed
  class ObservationChange : public Observation
  {

  public:

    //! Null constructor
    ObservationChange (); 

    //! Set the attributes that have been changed
    virtual void change (Observation* obs) const;

    //! Set the type of receiver feeds
    virtual void set_basis (Signal::Basis _basis);

    //! Set the state of the signal
    virtual void set_state (Signal::State _state);

    //! Set the source type
    virtual void set_type (Signal::Source _type);

    //! Set the dimension of each datum
    virtual void set_ndim (unsigned _ndim);

     //! Set the number of channels into which the band is divided
    virtual void set_nchan (unsigned _nchan);

    //! Set the number of polarizations
    virtual void set_npol (unsigned _npol);

    //! Set the number of bits per value
    virtual void set_nbit (unsigned _nbit);

    virtual void set_ndat (uint64_t _ndat);

    //! Set the telescope name
    virtual void set_telescope (const std::string& name);

    //! Set the receiver name
    virtual void set_receiver (const std::string& name);

    //! Set the source name
    virtual void set_source (const std::string& name);

    //! Set the coordinates of the source
    virtual void set_coordinates (sky_coord _coordinates);

    //! Set the dispersion measure recorded in the archive
    virtual void set_dispersion_measure (double dm);

    //! Set the rotation measure recorded in the archive
    virtual void set_rotation_measure (double dm);

    //! Set the centre frequency of the band-limited signal in MHz
    virtual void set_centre_frequency (double cf);

    //! Set the bandwidth of signal in MHz
    virtual void set_bandwidth (double _bandwidth);

    //! Set the start time of the leading edge of the first time sample
    virtual void set_start_time (MJD _start_time);

    //! Set the sampling rate (time samples per second in Hz)
    virtual void set_rate (double _rate);

    //! Set the amount by which data has been scaled
    virtual void set_scale (double _scale);

    //! Set true if frequency channels are out of order (band swappped)
    virtual void set_swap (bool _swap);

    //! Set true if the data are dual sideband
    virtual void set_dual_sideband (bool _dual);

    //! Set true if centre channel is centred on centre frequency
    virtual void set_dc_centred (bool _dc_centred);

    //! Set the observation identifier
    virtual void set_identifier (const std::string& _identifier);

    //! Set the instrument used to record signal
    virtual void set_machine (const std::string& _machine);

    //! Set the observation mode
    virtual void set_mode (const std::string& _mode);

    //! Set the cal frequency
    virtual void set_calfreq (double _calfreq);

  protected:

    bool telescope_changed;
    bool receiver_changed;
    bool source_changed;
    bool centre_frequency_changed;
    bool bandwidth_changed;
    bool type_changed;
    bool state_changed;
    bool basis_changed;
    bool rate_changed;
    bool start_time_changed;
    bool scale_changed;
    bool swap_changed;
    bool dc_centred_changed;
    bool identifier_changed;
    bool mode_changed;
    bool machine_changed;
    bool coordinates_changed;
    bool dispersion_measure_changed;
    bool rotation_measure_changed;
    bool ndat_changed;
    bool ndim_changed;
    bool nchan_changed;
    bool npol_changed;
    bool nbit_changed;
    bool calfreq_changed;
    bool dual_sideband_changed;

  };

}

#endif // ! __Observation_h
