//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/dsp.h,v $
   $Revision: 1.1 $
   $Date: 2003/07/28 13:40:46 $
   $Author: wvanstra $ */

#ifndef __baseband_dsp_h
#define __baseband_dsp_h

/*! \mainpage 
 
  \section intro Introduction
 
  The Baseband Data Reduction Library implements a family of C++
  classes that may be used in the loading and manipulation of
  observational data, primarily as a regularly sampled function of
  time.  This includes both phase-coherent data, as stored by baseband
  recording systems, and detected data, as produced by a filterbank
  system.  The functionality, contained in the dsp namespace, is
  divided into three main classes: data containers, operations,
  and auxilliary objects.

  The most general data container is the dsp::TimeSeries class, which
  is used to store the floating point representation of the signal in
  a variety of states.  The dsp::BitSeries class is used to store the
  N-bit digitized data before unpacking into a TimeSeries object.  The
  dsp::Loader class and its children are used to load data into the
  dsp::TimeSeries container.

  The main DSP algorithms are implemented by dsp::Operation and its
  sub-classes.  These operate on dsp::TimeSeries and can:
  <UL>
  <LI> convert digitized data to floating points (dsp::Unpack class)
  <LI> coherently dedisperse data (dsp::Convolution class)
  <LI> fold data using polyco (dsp::Fold class)
  <LI> etc...
  </UL>

  The auxilliary classes perform operations on arrays of data, such as
  multiplying a frequency response matrix by a spectrum field vector
  (e.g. the dsp::Response class).

 */

//! Contains all Baseband Data Reduction Library classes
namespace dsp {

  extern bool psrdisp_compatible;

}

#endif







