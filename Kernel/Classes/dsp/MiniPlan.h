//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __MiniPlan_h_
#define __MiniPlan_h_

#include "dsp/TimeSeries.h"
#include "dsp/Header.h"
#include "dsp/Printable.h"

/*!

A MiniPlan consists of a vector of SubInts and common information.

Each SubInt contains just enough information to
miniaturise/deminiaturise data, and also stores how that SubInt was
generated.  (i.e. when the data used to generate it started and how
much of that data was used.)

The MiniPlan also stores the number of bits for miniaturising and a
'sigma_max' parameter.  This is used for setting the levels for
miniaturising in terms of how many standard deviations a datum is away
from the mean.  The MiniPlan stores the rate of the input data so that
the relevant SubInt can be extracted for a particular input time.

The MiniPlan also stores 'requested_duration' and
'requested_scan_samps'.  These determine how long any incoming SubInts
being push-backed onto the SubInt vector are to be
(requested_duration), and how many samples are needed to generate
their means and std devs (requested_scan_samps).

*/

namespace dsp {

  class MiniPlan : public Reference::Able {

  public:
    
    //! Static verbosity flag
    static bool verbose;

    class SubInt : public Printable {
    public:
      //! When the SubInt is valid from
      MJD start_time;
      //! For how many samples the SubInt is valid for
      uint64_t duration;
      //! The means for each chan/pol group
      std::vector<std::vector<float> > means;
      //! The std devs for each chan/pol group
      std::vector<std::vector<float> > sigmas;
      //! The start time of the data the SubInt was generated from
      MJD gen_start;
      //! How many samples were used to generate the SubInt
      uint64_t gen_samps;

      //! Initialise from a vector of chars
      virtual unsigned read_from_chars(std::vector<char>& info,unsigned offset);

    protected:
      //! Does nothing
      virtual std::vector<char> null_pad(std::vector<char>& to_pad){ return to_pad; }
      //! Worker function for read()
      virtual std::vector<char> read_in_chars(int fd);
      //! Worker function for write_string();
      virtual std::string info_string();
    };

    //! Default constructor
    MiniPlan();

    //! Copy constructor- copies instances
    MiniPlan(const MiniPlan& mp);

    //! Instantiate from a file
    MiniPlan(std::string plan_file);

    //! Clone
    virtual MiniPlan* clone() const { return new MiniPlan(*this); }

    //! Assignment operator
    virtual MiniPlan& operator=(const MiniPlan& mp);
    
    //! Copies everything but the subints vector
    virtual void copy_attributes(const MiniPlan& mp);
    
    //! Virtual destructor
    virtual ~MiniPlan();

    //! Returns the Header associated with the MiniPlan
    virtual Reference::To<Header> MiniPlan2Header();

    //! Initialises the MiniPlan from a Header
    virtual void Header2MiniPlan(Reference::To<Header> hdr);

    uint64_t get_requested_scan_samps() const { return requested_scan_samps; }
    uint64_t get_requested_duration() const { return requested_duration; }
    float get_sigma_max() const { return sigma_max; }
    double get_rate() const { return rate; }
    unsigned get_nchan() const;
    unsigned get_npol() const;
    unsigned get_nsub() const;
    MJD get_start_time() const;
    MJD get_start_time(unsigned isub) const;
    MJD get_end_time() const;
    MJD get_end_time(unsigned isub) const;
    
    virtual void set_requested_scan_samps(uint64_t _requested_scan_samps);
    virtual void set_requested_duration(uint64_t _requested_duration);
    virtual void set_sigma_max(float _sigma_max){ sigma_max = _sigma_max; }

    //! Returns the lower threshold- points below this are lopped off
    virtual float get_lower_threshold(unsigned ichan, unsigned ipol, unsigned isub);
    
    //! Returns the upper threshold- points above this are lopped off
    virtual float get_upper_threshold(unsigned ichan, unsigned ipol, unsigned isub);
   
    //! Returns the step between digitization levels
    virtual float get_step(unsigned ichan, unsigned ipol, unsigned isub,
			   unsigned nbit);

    //! Add in subint sets for the given TimeSeries
    //! Each TimeSeries this is called on must be contiguous with the last
    //! Returns the number of samples that were added in to the plan
    //! (This is between 0 and data->get_ndat())
    virtual uint64_t add_data(const TimeSeries* data);

    //! Stretch the last subint stored until it has reached the maximum
    //! number of samples it is allowed to cover (which is derived from
    //! 'requested_duration')
    virtual uint64_t stretch_last_subint(const TimeSeries* data);

    //! Append 'miniplan's subints onto the end of this
    virtual void append(Reference::To<MiniPlan> miniplan);

    //! Create a new MiniPlan with subints covering the range start->end
    virtual Reference::To<MiniPlan> extract(MJD start, MJD end);

    //! Makes the final subints 'ext' samples longer in duration
    virtual void extend_last_subint(uint64_t ext);

    //! Extends the 'duration' attribute of the last SubInt so that it ends at this time
    virtual void extend(MJD _end_time);

    //! Extends the first subint backwards in time to start at this time
    virtual void backwards_extend(MJD _start_time);

    //! Hack for debugging.  Should be taken out once MiniPlan is working
    std::vector<SubInt>& get_subints(){ return subints; }
    
  protected:

    /*
    //! Worker function for read()
    virtual vector<char> read_in_chars(int fd);
    */

    //! Used by constructors
    virtual void init();

    //! Checks that requested_duration and requested_scan_samps are valid;
    virtual void check_requests(const TimeSeries* data);
    
    /*
    //! Returns a string for print() to print out
    virtual string info_string();
    */

    /*
    //! Pads up the given vector with null characters and returns it
    //! Called by write_chars() which is called by write()
    //! Does nothing for MiniPlan class- null characters are written by the Header it writes out
    virtual vector<char> null_pad(vector<char>& to_pad);
    */

    /*
    //! Prints out the subint
    virtual string print_subint(unsigned isub);
    */    

    /*
    //! Initialises the subints vector from the 'subint_lines' strings
    //! Returns the number of lines read
    virtual unsigned read_subint_lines(vector<string>& subint_lines,unsigned nsub,
				       unsigned nchan, unsigned npol);
    */

    //! Scan the attributes stored in the Header in
    virtual void read_header(Reference::To<Header> hdr,unsigned& nsub,
			     unsigned& nchan, unsigned& npol);
    
    //! Returns a Header associated with the non-SubInt attributes
    virtual Reference::To<Header> get_header();

    //! Add in a SubInt for the given TimeSeries
    virtual void add_subint(const TimeSeries* data,uint64_t& offset);

    virtual void set_rate(double _rate){ rate = _rate; }

    //! The SubInts
    std::vector<SubInt> subints;

  private:

    //! Returns true if the 'means' and 'sigmas' vectors in the two SubInts are identical
    bool equal_coeffs(SubInt& s1, SubInt& s2);

    //! Helper utility to check ichan, ipol, isub are valid
    void bound_checks(unsigned ichan, unsigned ipol, unsigned isub);

    /*
    //! Worker function for read_in_chars
    unsigned parse_param(const vector<string>& lines,string param);
    */

    //! Requested number of samples to generate mean etc. from (0 for all) [0]
    uint64_t requested_scan_samps;

    //! Requested number of samples to pump into a single subint (0 for as big as the first TimeSeries added in.) [0]
    uint64_t requested_duration;

    //! The (positive) cutoff for digitisation.  Any samples higher than this will be lopped down to size [8.0];
    //! This number determines the step between levels.
    //! e.g. if sigma_max==4.0 and nbit==2 then levels will be (in sigma): -3.0, -1.0, 1.0, 3.0
    //! So data in range [-4,-2) -> -3 (00)
    //! So data in range [-2,0)  -> -1 (01)
    //! So data in range [0,2)   ->  1 (10)
    //! So data in range [2,4)   ->  3 (11)
    //! So data outside these ranges gets lopped down to -3 or 3.
    float sigma_max;

    //! The rate of the data being digitised
    double rate;
    
  };

}

#endif

