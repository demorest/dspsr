//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/General/dsp/Shape.h,v $
   $Revision: 1.8 $
   $Date: 2003/12/11 18:36:03 $
   $Author: wvanstra $ */

#ifndef __Shape_h
#define __Shape_h

#ifdef ACTIVATE_MPI
#include <mpi.h>
#endif

#include "ReferenceAble.h"

namespace dsp {

  //! Base class of objects that Shape data in the time or frequency domain
  class Shape : public Reference::Able {

  public:
    static bool verbose;

    //! Default constructor
    Shape ();
    //! Destructor
    virtual ~Shape ();
    //! Copy constructor
    Shape (const Shape&);
    //! Assignment operator
    const Shape& operator = (const Shape&);

    //! Set the dimensions of the data
    virtual void resize (unsigned npol, unsigned nchan,
			 unsigned ndat, unsigned ndim);

    //! Get the number of polarizations
    unsigned get_npol()  const { return npol; }

    //! Get the number of frequency channels
    unsigned get_nchan() const { return nchan; }

    //! Get the number of datum in each of the nchan*npol divisions
    unsigned get_ndat()  const { return ndat; }

    //! Get the dimension of each datum (e.g. 2=complex 8=Jones)
    unsigned get_ndim()  const { return ndim; }

    //! Returns true if the npol, nchan, and ndat dimensions match
    virtual bool matches (const Shape* shape);

    //! Scrunch each dimension to a new ndat
    void scrunch_to (unsigned ndat);

    //! Rotate data so that Shape[i] = Shape[i+npt]
    void rotate (int npt);

    //! Set all values to zero
    void zero ();

    //! Borrow the data from the specified channel of another Shape
    void borrow (const Shape&, unsigned ichan=0);
   
    //! Divide each point by factor
    const Shape& operator /= (float factor);

    //! Multiply each point by factor
    const Shape& operator *= (float factor);

    //! Add another Shape to this one
    const Shape& operator += (const Shape&);

    //! Provide access to the data for the specified polarization
    float* get_datptr (unsigned ichan, unsigned ipol) {
      return buffer + offset * ipol + ndat*ndim * ichan;
    }

    //! Provide access to the data for the specified polarization
    const float* get_datptr (unsigned ichan, unsigned ipol) const {
      return buffer + offset * ipol + ndat*ndim * ichan;
    }

  protected:

    //! Data points
    float*   buffer;

    //! Size of the data buffer
    unsigned bufsize;

    //! Offset between datum from each polarization
    unsigned offset;

    //! Number of polarizations
    unsigned npol;

    //! Number of frequency divisions (channels)
    unsigned nchan;

    //! Number of datum in each of the npol*nchan divisions
    unsigned ndat;

    //! Dimension of each datum
    unsigned ndim;

    //! Flag that data are borrowed from another Shape
    bool borrowed;

    void init ();
    void size_dataspace ();
    void destroy ();

#if PGPLOT
    void plot (float centre, float width, const char* label, 
		bool swap=false, int dimension=0, bool one_poln=false);
#endif

  };

}

// these are declared outside of the namespace, to simplify their
// use in the stdmpi templates
#ifdef ACTIVATE_MPI
int mpiPack_size (const dsp::Shape&,
		  MPI_Comm comm, int* size);
int mpiPack      (const dsp::Shape&, void* outbuf,
		  int outcount, int* position, MPI_Comm comm);
int mpiUnpack    (void* inbuf, int insize, int* position, 
		  dsp::Shape*, MPI_Comm comm);
#endif

#endif // #ifndef __Shape_h
