//-*-C++-*-

#ifndef __MiniPlan_h_
#define __MiniPlan_h_

#include <vector>
#include <string>

#include "Reference.h"
#include "environ.h"
#include "MJD.h"
#include "Printable.h"

#include "dsp/Observation.h"
#include "dsp/TimeSeries.h"

/*!

  Each channel in a dsp::TimeSeries can be divided up short segments
  of data- subints.  Each of these subints can be digitized separately.

*/

namespace dsp {

  class MiniPlan : public Printable {

  public:
    
    class SubInt : public Reference::Able {
    public:
      SubInt();
      SubInt(string line);
      virtual ~SubInt();

      uint64 get_duration() const { return duration; }
      MJD get_start_time() const { return start_time; }
      float get_mean() const { return mmean; }
      float get_sigma() const { return ssigma; }
      
      void set_duration(uint64 _duration);
      void set_start_time(MJD _start_time){ start_time = _start_time; }
      void set_mean(float _mean){ mmean = _mean; }
      void set_sigma(float _sigma){ ssigma = _sigma; }

      string print();

      //! Returns true if the given MJD is between start_time and duration
      bool covers(MJD time,double rate);

      //! Extends duration by 'ext'
      void extend(uint64 ext);

    private:
      uint64 duration; // Must divide 8
      MJD start_time;
      float mmean;
      float ssigma;
    };
    
    class SubIntVector : public Reference::Able {
    public:
      friend class MiniPlan;

      SubIntVector();
      virtual ~SubIntVector();

      virtual void add_subint(Reference::To<SubInt> _subint);
      
      //! Shares the references to the subints 
      virtual void share(SubIntVector& has_subints);

      unsigned size();

      string print_subint(unsigned isub);

    protected:
      void resize(unsigned _size);

      vector<Reference::To<SubInt> > subints;
    };

    //! Default constructor
    MiniPlan();

    //! Copy constructor- copies references, but not instances
    MiniPlan(const MiniPlan& mp);

    //! Virtual destructor
    virtual ~MiniPlan();

    uint64 get_requested_scan_samps() const { return requested_scan_samps; }
    uint64 get_requested_duration() const { return requested_duration; }
    float get_sigma_max() const { return sigma_max; }

    void set_requested_scan_samps(uint64 _requested_scan_samps);
    void set_requested_duration(uint64 _requested_duration);
    void set_sigma_max(float _sigma_max){ sigma_max = _sigma_max; }

    //! Returns the number of subints stored in each SubIntVector
    unsigned get_nsubints();

    //! Returns the number of samples (floats) subint 'isub' is past the start
    inline uint64 get_ts_offset(unsigned isub);
   
    //! Returns the duration of this particular subint
    inline uint64 get_duration(unsigned isub);
    
    //! Returns the number of bytes in the output BitSeries this particular subint starts at
    inline uint64 get_bs_offset(unsigned ichan, unsigned ipol, unsigned isub, unsigned nbit);

    //! Returns the number of bytes in the output MiniSeries this particular subint starts at
    inline uint64 get_ms_offset(unsigned isub, unsigned nbit);

    //! Returns the lower threshold- points below this are lopped off
    inline float get_lower_threshold(unsigned ichan, unsigned ipol, unsigned isub, unsigned nbit);

    //! Returns the upper threshold- points above this are lopped off
    inline float get_upper_threshold(unsigned ichan, unsigned ipol, unsigned isub, unsigned nbit);
   
    //! Returns the step between digitization levels
    inline float get_step(unsigned ichan, unsigned ipol, unsigned isub, unsigned nbit);

    //! Add in subint sets for the given TimeSeries
    //! Each TimeSeries this is called on must be contiguous with the last
    virtual void add_data(const TimeSeries* data);

    //! Retrieve a reference to the info
    Reference::To<Observation> get_info(){ return info; }

    //! Read in the MiniPlan from a character array
    unsigned read_from_chars( vector<char>& info, unsigned offset);

    //! Create a new MiniPlan with subints covering the range start->end
    Reference::To<MiniPlan> extract(MJD start, MJD end, double rate);

    //! Resize the subints_vector
    void resize(unsigned nchan, unsigned npol);

    unsigned get_nchan(){ return subint_vectors.size(); }
    unsigned get_npol(){ return subint_vectors[0].size(); }

    //! Makes the final subints 'ext' samples longer in duration
    void extend_last_subint(uint64 ext);

    //! Checks that requested_duration and requested_scan_samps are valid;
    void check_requests(const TimeSeries* data);

    //! Returns the end time of the final subint
    MJD get_end_time(double rate);

  protected:
    
    //! Called by read() to read in the MiniPlan from a file descriptor
    void read_info(int fd);

    //! Called by write() to write the MiniPlan to disk
    void write_info(int fd);

    //! Returns a string for print() to print out
    string info_string();

    //! Add in a subint set for the given TimeSeries
    virtual void add_subint_set(const TimeSeries* data,uint64 offset);

    //! Helper method for read_info() and read_from_chars()
    void parse_lines(unsigned nchan, unsigned npol, unsigned nsub,
		     const vector<string>& lines, unsigned line_offset);

    //! Returns the subint that covers 'time'.  Returns -1 for failure
    int get_covering_subint(MJD time, double rate);
    
    //! Requested number of samples to generate mean etc. from (0 for all) [0]
    uint64 requested_scan_samps;

    //! Requested number of samples to pump into a single subint (0 for as big as the first TimeSeries added in.) [-1]
    uint64 requested_duration;

    //! The (positive) cutoff for digitisation.  Any samples higher than this will be lopped down to size [8.0];
    //! This number determines the step between levels.
    //! e.g. if sigma_max==4.0 and nbit==2 then levels will be (in sigma): -3.0, -1.0, 1.0, 3.0
    //! So data in range [-4,-2) -> -3 (00)
    //! So data in range [-2,0)  -> -1 (01)
    //! So data in range [0,2)   ->  1 (10)
    //! So data in range [2,4)   ->  3 (11)
    //! So data outside these ranges gets lopped down to -3 or 3.
    float sigma_max;

    //! Each chan/pol has a SubIntVector 
    vector<vector<Reference::To<SubIntVector> > > subint_vectors;

    //! The last set of data added in
    Reference::To<Observation> info;

  };

}

#endif





