//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Response.h,v $
   $Revision: 1.19 $
   $Date: 2003/01/07 15:46:21 $
   $Author: wvanstra $ */

#ifndef __Response_h
#define __Response_h

#include <vector>
#include <complex>
#include <string>

#include "dsp/Shape.h"
#include "Jones.h"

namespace dsp {

  class Observation;

  class Response : public Shape {

  public:

    //! Maximum ndat allowed
    static unsigned ndat_max;

    //! null constructor
    Response();

    //! Get the size of the positive half of the impulse response, \f$m_+\f$
    /*! Get the number of complex time samples in the t>0 half of the
      corresponding impulse response function */
    unsigned get_impulse_pos () const { return impulse_pos; }

    //! Get the size of the negative half of the impulse response, \f$m_-\f$
    /*! Get the number of complex time samples in the t<0 half of the
      corresponding impulse response function */
    unsigned get_impulse_neg () const { return impulse_neg; }

    //! Set the size of the positive half of the impulse response, \f$m_+\f$
    /*! Set the number of complex time samples in the t>0 half of the
      corresponding impulse response function */
    void set_impulse_pos (unsigned _impulse_pos) { impulse_pos =_impulse_pos; }

    //! Set the size of the negative half of the impulse response, \f$m_-\f$
    /*! Set the number of complex time samples in the t<0 half of the
      corresponding impulse response function */
    void set_impulse_neg (unsigned _impulse_neg) { impulse_neg =_impulse_neg; }

    //! Set the flag for a bin-centred spectrum
    void set_dc_centred (bool dc_centred);

    //! Get the flag for a bin-centred spectrum
    bool get_dc_centred () const { return dc_centred; }

    //! Return the minimum useable ndat
    unsigned get_minimum_ndat () const;

    //! Resize with ndat set to the optimal value
    void set_optimal_ndat ();

    //! Given impulse_pos and impulse_neg, check that ndat is large enough
    void check_ndat () const;

    //! Construct frequency response from complex phasors
    void set (const vector<complex<float> >& phasors);

    //! Construct frequency response from jones matrices
    void set (const vector<Jones<float> >& jones);

    //! Multiply spectrum by complex frequency response
    void operate (float* spectrum, unsigned poln=0, int ichan=-1);

    //! Multiply spectrum vector by complex matrix frequency response
    void operate (float* spectrum1, float* spectrum2, int ichan=-1);

    //! Integrate the power of spectrum into self
    void integrate (float* spectrum, unsigned poln=0, int ichan=-1);

    //! Integrate coherency matrix of vector spectrum into self
    void integrate (float* spectrum1, float* spectrum2, int ichan=-1);

    //! Match the frequency response to the input Observation
    virtual void match (const Observation* input, unsigned channels=0);

    //! Match the frequency response to another Response
    virtual void match (const Response* response);

    //! Returns true if the dimension and ordering match
    virtual bool matches (const Response* response);

    //! Modify the out Observation information as seen fit by sub-classes
    virtual void mark (Observation* output);

    //! Re-organize frequency bins to reflect natural ordering (DC->Nyq)
    void naturalize ();

    //! Enable Response to be used in Transformation template
    virtual bool state_is_valid (string& reason) { return true; }

  protected:

    //! Swap halves of bandpass(es)
    void swap (bool each_chan = true);

    //! Complex time samples in t>0 half of corresponding impulse response
    unsigned impulse_pos;

    //! Complex time samples in t<0 half of corresponding impulse response
    unsigned impulse_neg;

    //! Toggled every time Response::swap(false) is called (default: false)
    bool whole_swapped;

    //! Toggled every time Response::swap(true) is called (default: false)
    bool chan_swapped;

    //! Toggled when built for a bin-centred spectrum
    bool dc_centred;

  };

}

#endif


