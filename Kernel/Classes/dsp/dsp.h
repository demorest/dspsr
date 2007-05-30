//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/dsp.h,v $
   $Revision: 1.5 $
   $Date: 2007/05/30 07:35:41 $
   $Author: straten $ */

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

  \section backend Adding a new File Format

  When adding a new file format, the following steps should be followed:
  <UL>

  <LI> Create a new subdirectory of baseband, say baseband/backend,
  and create all new files here.

  <LI> Inherit dsp::File or one of its derived classes, implement the
  header-parsing code, and add the new class to File_registry.C using
  preprocessor directives.

  <LI> Inherit dsp::Unpacker or one of its derived classes, implement
  the bit-unpacking code, and add the new class to Unpacker_registry.C
  using preprocessor directives.

  <LI> Define the appropriate preprocessor directives within an
  optional section of Makefile.backends.

  </UL>

 */

//! Contains all Baseband Data Reduction Library classes
namespace dsp {

  //! Set true to enable backward compatibility features
  extern bool psrdisp_compatible;

  //! The baseband/dsp version number
  extern const float version;

  //! Set the verbosity level of various base classes
  void set_verbosity (unsigned level);

}

#endif







