/***************************************************************************
 *
 *   Copyright (C) 2011, 2013 by James M Anderson  (MPIfR)
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LuMPUnpacker.h"
#include "dsp/LuMPFile.h"

#include "Error.h"

#include <assert.h>
#include "lump_internals.h"
#include "MPIfR_Real16_t.h"
#include <cstdio>


namespace {
//! 16 element array to provide conversion from 4 bit two's complement
//  integers to floating point.
const float int4_t_values_Real32_t[16] = {
    +0.0f, +1.0f, +2.0f, +3.0f, +4.0f, +5.0f, +6.0f, +7.0f,
    -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f};
}



using namespace std;

//! Constructor
dsp::LuMPUnpacker::LuMPUnpacker (const char* name) : Unpacker (name)
{
    return;
}

//! Return true if the unpacker support the specified output order
bool dsp::LuMPUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::LuMPUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::LuMPUnpacker::matches (const Observation* observation)
{
  unsigned nbit = observation->get_nbit();
  unsigned npol = observation->get_npol();
  unsigned ndim = observation->get_ndim();

  const LuMPObservation* lump_obs
    = dynamic_cast<const LuMPObservation*> (observation);

  if (!lump_obs)
  {
    if (verbose)
      cerr << "dsp::LuMPUnpacker::matches"
	" Observation is not a LuMPObservation" << endl;
    return false;
  }

  dsp::BinaryFormat binary_format = lump_obs->get_binary_format();
  dsp::DataEndianness data_endianness = lump_obs->get_data_endianness();
  dsp::PolOrdering pol_ordering = lump_obs->get_pol_ordering();

  const Signal::State state = observation->get_state();

  if (verbose)
      std::cerr << "dsp::LuMPUnpacker::matches machine=" << observation->get_machine()
                << " nbit=" << nbit << " npol=" << npol << " ndim=" << ndim
                << " bin_form=" << int(binary_format) << " endian=" << int(data_endianness)
                << " pol_ord=" << int(pol_ordering)
                << " state=" << int(state) << endl;
  if (verbose)
      std::cerr << "General information in dsp::LuMPUnpacker::matches"
                << " ndat=" << observation->get_ndat()
                << " telescope=" << observation->get_telescope()
                << " source=" << observation->get_source()
                << " rate=" << observation->get_rate()
                << " nbytes=" << observation->get_nbytes()
                << " nbyte=" << observation->get_nbyte()
                << " nsamples(32)=" << observation->get_nsamples(32)
                << " dispersion_measure=" << observation->get_dispersion_measure()
          //<< " =" << observation->get_()
          //<< " =" << observation->get_()
                << endl;
  if (verbose) {
      char s[32];
      std::snprintf(s,32,"%.5f",observation->get_rate());
      std::cerr << "General information in dsp::LuMPUnpacker::matches"
                << " rate=" << s << endl;
  }
  
  if(ndim > 2) {
      std::cerr << "nidm > 2 not supported for LuMP" << endl;
      return false;
  }
  else if((ndim == 2) && (npol > 2)) {
      std::cerr << "DSPSR does not properly support complex valued full Stokes information, but the input file is requesting ndim==2 and npol > 2.  If you have correlator data in Jones matrix order (such as RR RL LR LL), then you will have to evaluate your visibilities for a specific direction on the sky and convert to real-valued data first." << endl;
  }

  bool endian_test = (data_endianness == dsp::DataLittleEndian) || (data_endianness == dsp::DataBigEndian);
  bool pol_test_0 = (pol_ordering == dsp::DSPSROrdering) || (pol_ordering == dsp::JonesMatrixOrdering);
  bool pol_test_1 = (npol == 1) || (npol == 2) || (npol == 4);
  bool state_test = (state == Signal::Nyquist) || (state == Signal::Analytic)
                    || (state == Signal::Intensity) || (state == Signal::PPQQ)
                    || (state == Signal::Coherence) || (state == Signal::Stokes);
  bool bit_format_test = (  (nbit==4 || nbit==8 || nbit==16 )
                         && (binary_format == dsp::IntegerBinForm)  )
                         || (  (nbit==16 || nbit==32 || nbit==64 )
                            && (binary_format == dsp::IEEE_FloatBinForm)  );
  bool dim_test_0 = (ndim == 1) || (ndim == 2);
  bool dim_test_1 = ( (ndim == 1) && (nbit >= 8) )
                    || (ndim == 2);
  bool dim_test_2 = ( (npol == 1)
                    || (npol == 2)
                    || ( (npol == 4) && (ndim == 1) ) );
  bool machine_test = observation->get_machine() == "LuMP";

  bool combined_test = machine_test && endian_test && pol_test_0 && pol_test_1
                       && state_test && bit_format_test && dim_test_0
                       && dim_test_1 && dim_test_2;
  return combined_test;
  
  // return observation->get_machine() == "LuMP"
  //     && ((data_endianness == dsp::DataLittleEndian) || (data_endianness == dsp::DataBigEndian))
  //     && ((pol_ordering == dsp::DSPSROrdering) || (pol_ordering == dsp::JonesMatrixOrdering))
  //     && ((ndim == 1) || (ndim == 2))
  //     && ((npol == 1) || (npol == 2) || (npol == 4))
  //     && ((state == Signal::Nyquist) || (state == Signal::Analytic)
  //        || (state == Signal::Intensity) || (state == Signal::PPQQ)
  //        || (state == Signal::Coherence) || (state == Signal::Stokes))
  //     && ( ((nbit==4 || nbit==8 || nbit==16 ) && (binary_format == dsp::IntegerBinForm))
  //        || ((nbit==16 || nbit==32 || nbit==64 ) && (binary_format == dsp::IEEE_FloatBinForm)) );
}


// Return the polarization position associated with the specified digitizer.
unsigned dsp::LuMPUnpacker::get_output_ipol (unsigned idig) const
{
  return idig;
}

// Return the channel number for the specified digitizer.  Since LOFAR
// does not have different digitizers for different channels, return 0.
unsigned dsp::LuMPUnpacker::get_output_ichan (unsigned idig) const
{
  return 0;
}

void dsp::LuMPUnpacker::unpack ()
{
  const uint64_t ndat = input->get_ndat();
  const uint_fast32_t nchan = input->get_nchan();
  const uint_fast32_t ndim = input->get_ndim();
  const uint_fast32_t npol = input->get_npol();
  const uint_fast32_t nbit = input->get_nbit();
  const Signal::State state = input->get_state();

  const Observation* obs = input->get_loader()->get_info();
  const LuMPObservation* lump = dynamic_cast<const LuMPObservation*>(obs);
  assert (lump != NULL);

  const dsp::BinaryFormat binary_format = lump->get_binary_format();
  const dsp::DataEndianness data_endianness = lump->get_data_endianness();
  const dsp::PolOrdering pol_ordering = lump->get_pol_ordering();

  bool NEED_TO_BYTESWAP = ( ((data_endianness == dsp::DataLittleEndian) && (!MACHINE_LITTLE_ENDIAN))
                          || ((data_endianness == dsp::DataBigEndian) && (MACHINE_LITTLE_ENDIAN)) );

  // declare things here which cause problems with jumps in the switch statements
  // below crossing initializations.
  unsigned ipol_map[4] = {0,1,2,3};
  uint_fast64_t stride = 1;
  uint_fast64_t NUM_POINTS = 0;
  const uint8_t* restrict from_uint8_t   = 0;
  const int8_t* restrict from_int8_t     = 0;
  const int16_t* restrict from_int16_t   = 0;
  const int32_t* restrict from_int32_t   = 0;
  const int64_t* restrict from_int64_t   = 0;
  const uint16_t* restrict from_uint16_t = 0;
  const MPIfR::DATA_TYPE::Real16_t* restrict from_Real16_t   = 0;
  const float* restrict from_float       = 0;
  const double* restrict from_double     = 0;
  float* restrict into = 0;
  MPIfR::DATA_TYPE::Real16_t X0, X1, X2, X3;


  
  switch ( output->get_order() )
  {
  case TimeSeries::OrderFPT:
      if(!NEED_TO_BYTESWAP){
          switch(nbit) {
          case 4: // nbit == 4
              if(ndim == 1) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "nbit == 4 and ndim == 1 not allowed for LuMP");
              }
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 4 bits");
              }
              from_uint8_t = reinterpret_cast<const uint8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim) / 2;
                  switch(ndim) {
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  // LOFAR sticks the real part into the lower
                                  // 4 bits, and the imaginary part into the
                                  // upper 4 bits.
                                  uint8_t u = from_uint8_t[offset];
                                  *into++ = int4_t_values_Real32_t[u & 0xF];
                                  *into++ = int4_t_values_Real32_t[u >> 4];
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 8: // nbit == 8
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 8 bits");
              }
              from_int8_t = reinterpret_cast<const int8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  switch(ndim) {
                  case 1:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_int8_t[offset]);
                              }
                          }
                      }
                      break;
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_int8_t[offset+0]);
                                  *into++ = float(from_int8_t[offset+1]);
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 16: // nbit == 16
              switch(binary_format) {
              case dsp::IntegerBinForm:
                  from_int16_t = reinterpret_cast<const int16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                  case 1:
                  case 2:
                      if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          ipol_map[0] = 0;
                          ipol_map[1] = 2;
                          ipol_map[2] = 3;
                          ipol_map[3] = 1;
                      }
                      else {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          ipol_map[0] = 0;
                          ipol_map[1] = 1;
                          ipol_map[2] = 2;
                          ipol_map[3] = 3;
                      }
                      stride = uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      switch(ndim) {
                      case 1:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      *into++ = float(from_int16_t[offset]);
                                  }
                              }
                          }
                          break;
                      case 2:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      *into++ = float(from_int16_t[offset+0]);
                                      *into++ = float(from_int16_t[offset+1]);
                                  }
                              }
                          }
                          break;
                      default:
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unsupported ndim");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  }
                  break;
              case dsp::IEEE_FloatBinForm: // binary16 floating point format
                  from_Real16_t = reinterpret_cast<const MPIfR::DATA_TYPE::Real16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                  case 1:
                  case 2:
                      if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          ipol_map[0] = 0;
                          ipol_map[1] = 2;
                          ipol_map[2] = 3;
                          ipol_map[3] = 1;
                      }
                      else {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          ipol_map[0] = 0;
                          ipol_map[1] = 1;
                          ipol_map[2] = 2;
                          ipol_map[3] = 3;
                      }
                      stride = uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      switch(ndim) {
                      case 1:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      *into++ = float(from_Real16_t[offset].to_Real32_t());
                                  }
                              }
                          }
                          break;
                      case 2:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      *into++ = float(from_Real16_t[offset+0].to_Real32_t());
                                      *into++ = float(from_Real16_t[offset+1].to_Real32_t());
                                  }
                              }
                          }
                          break;
                      default:
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unsupported ndim");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 16 bits");
              } // switch binary_format
              break;
          case 32: // nbit == 32
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 32 bits");
              }
              from_float = reinterpret_cast<const float* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  switch(ndim) {
                  case 1:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_float[offset]);
                              }
                          }
                      }
                      break;
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_float[offset+0]);
                                  *into++ = float(from_float[offset+1]);
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 64: // nbit == 64
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 64 bits");
              }
              from_double = reinterpret_cast<const double* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  switch(ndim) {
                  case 1:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_double[offset]);
                              }
                          }
                      }
                      break;
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_double[offset+0]);
                                  *into++ = float(from_double[offset+1]);
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          default:
              throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                           "unrecognized nbit");
          } // switch nbit
          break;
      }
      else { // NEED_TO_BYTESWAP
          into = reinterpret_cast<float* restrict>(output->get_dattfp());
          switch(nbit) {
          case 4: // nbit == 4
              // no actual need to byteswap here, but maintained for code simplicity
              if(ndim == 1) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "nbit == 4 and ndim == 1 not allowed for LuMP");
              }
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 4 bits");
              }
              from_uint8_t = reinterpret_cast<const uint8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim) / 2;
                  switch(ndim) {
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  // LOFAR sticks the real part into the lower
                                  // 4 bits, and the imaginary part into the
                                  // upper 4 bits.
                                  uint8_t u = from_uint8_t[offset];
                                  *into++ = int4_t_values_Real32_t[u & 0xF];
                                  *into++ = int4_t_values_Real32_t[u >> 4];
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 8: // nbit == 8
              // no actual need to byteswap here, but maintained for code simplicity
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 8 bits");
              }
              from_int8_t = reinterpret_cast<const int8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  switch(ndim) {
                  case 1:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_int8_t[offset]);
                              }
                          }
                      }
                      break;
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = float(from_int8_t[offset+0]);
                                  *into++ = float(from_int8_t[offset+1]);
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 16: // nbit == 16
              switch(binary_format) {
              case dsp::IntegerBinForm:
                  from_int16_t = reinterpret_cast<const int16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                  case 1:
                  case 2:
                      if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          ipol_map[0] = 0;
                          ipol_map[1] = 2;
                          ipol_map[2] = 3;
                          ipol_map[3] = 1;
                      }
                      else {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          ipol_map[0] = 0;
                          ipol_map[1] = 1;
                          ipol_map[2] = 2;
                          ipol_map[3] = 3;
                      }
                      stride = uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      switch(ndim) {
                      case 1:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      *into++ = float(byteswap_int16_t(from_int16_t[offset]));
                                  }
                              }
                          }
                          break;
                      case 2:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      *into++ = float(byteswap_int16_t(from_int16_t[offset+0]));
                                      *into++ = float(byteswap_int16_t(from_int16_t[offset+1]));
                                  }
                              }
                          }
                          break;
                      default:
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unsupported ndim");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  }
                  break;
              case dsp::IEEE_FloatBinForm: // binary16 floating point format
                  from_uint16_t = reinterpret_cast<const uint16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                  case 1:
                  case 2:
                      if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          ipol_map[0] = 0;
                          ipol_map[1] = 2;
                          ipol_map[2] = 3;
                          ipol_map[3] = 1;
                      }
                      else {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          ipol_map[0] = 0;
                          ipol_map[1] = 1;
                          ipol_map[2] = 2;
                          ipol_map[3] = 3;
                      }
                      stride = uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      switch(ndim) {
                      case 1:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      X0.load_bits_byteswap(from_uint16_t[offset]);
                                      *into++ = X0.to_Real32_t();
                                  }
                              }
                          }
                          break;
                      case 2:
                          for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                                   poll_offset += ndim) {
                              for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                       chan_offset += npol*ndim) {
                                  uint_fast64_t offset = poll_offset + chan_offset;
                                  into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                                  for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                      X0.load_bits_byteswap(from_uint16_t[offset+0]);
                                      X1.load_bits_byteswap(from_uint16_t[offset+1]);
                                      *into++ = X0.to_Real32_t();
                                      *into++ = X1.to_Real32_t();
                                  }
                              }
                          }
                          break;
                      default:
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unsupported ndim");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 16 bits");
              } // switch binary_format
              break;
          case 32: // nbit == 32
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 32 bits");
              }
              from_int32_t = reinterpret_cast<const int32_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  switch(ndim) {
                  case 1:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[offset]));
                              }
                          }
                      }
                      break;
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[offset+0]));
                                  *into++ = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[offset+1]));
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 64: // nbit == 64
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 64 bits");
              }
              from_int64_t = reinterpret_cast<const int64_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
              case 1:
              case 2:
                  if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      ipol_map[0] = 0;
                      ipol_map[1] = 2;
                      ipol_map[2] = 3;
                      ipol_map[3] = 1;
                  }
                  else {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      ipol_map[0] = 0;
                      ipol_map[1] = 1;
                      ipol_map[2] = 2;
                      ipol_map[3] = 3;
                  }
                  stride = uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  switch(ndim) {
                  case 1:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[offset]));
                              }
                          }
                      }
                      break;
                  case 2:
                      for (uint_fast64_t ipol=0, poll_offset=0; ipol<npol; ipol++,
                               poll_offset += ndim) {
                          for (uint_fast64_t ichan=0, chan_offset=0; ichan<nchan; ichan++,
                                   chan_offset += npol*ndim) {
                              uint_fast64_t offset = poll_offset + chan_offset;
                              into = reinterpret_cast<float* restrict>(output->get_datptr (unsigned(ichan), ipol_map[ipol]));
                              for (uint_fast64_t n=0; n < ndat; ++n, offset += stride) {
                                  *into++ = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[offset+0]));
                                  *into++ = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[offset+1]));
                              }
                          }
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          default:
              throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                           "unrecognized nbit");
          } // switch nbit
          break;
      }
      break;
  case TimeSeries::OrderTFP:
      if(!NEED_TO_BYTESWAP){
          into = reinterpret_cast<float* restrict>(output->get_dattfp());
          switch(nbit) {
          case 4: // nbit == 4
              if(ndim == 1) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "nbit == 4 and ndim == 1 not allowed for LuMP");
              }
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 4 bits");
              }
              from_uint8_t = reinterpret_cast<const uint8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  switch(ndim) {
                  case 2:
                      // easy, just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                                   * uint_fast64_t(nchan)
                                   * uint_fast64_t(npol)
                                   * uint_fast64_t(ndim) / 2;
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          // LOFAR sticks the real part into the lower
                          // 4 bits, and the imaginary part into the
                          // upper 4 bits.
                          uint8_t u = from_uint8_t[i];
                          *into++ = int4_t_values_Real32_t[u & 0xF];
                          *into++ = int4_t_values_Real32_t[u >> 4];
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              case 4:
                  if(ndim == 1) {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "nbit == 4 and ndim == 1 not allowed for LuMP");
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 8: // nbit == 8
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 8 bits");
              }
              from_int8_t = reinterpret_cast<const int8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  // easy, just copy over directly
                  NUM_POINTS = uint_fast64_t(ndat)
                      * uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                      into[i] = float(from_int8_t[i]);
                  }
                  break;
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                    || (state == Signal::Stokes) ) {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = float(from_int8_t[i]);
                      }
                  }
                  else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(4);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                          into[i+0] = float(from_int8_t[i+0]);
                          into[i+1] = float(from_int8_t[i+3]);
                          into[i+2] = float(from_int8_t[i+1]);
                          into[i+3] = float(from_int8_t[i+2]);
                      }
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized pol_ordering and state combination");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 16: // nbit == 16
              switch(binary_format) {
              case dsp::IntegerBinForm:
                  from_int16_t = reinterpret_cast<const int16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 1:
                  case 2:
                      // easy, just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = float(from_int16_t[i]);
                      }
                      break;
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                      if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                        || (state == Signal::Stokes) ) {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(npol)
                              * uint_fast64_t(ndim);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                              into[i] = float(from_int16_t[i]);
                          }
                      }
                      else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(4);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                              into[i+0] = float(from_int16_t[i+0]);
                              into[i+1] = float(from_int16_t[i+3]);
                              into[i+2] = float(from_int16_t[i+1]);
                              into[i+3] = float(from_int16_t[i+2]);
                          }
                      }
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unrecognized pol_ordering and state combination");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  }
                  break;
              case dsp::IEEE_FloatBinForm: // binary16 floating point format
                  from_Real16_t = reinterpret_cast<const MPIfR::DATA_TYPE::Real16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 1:
                  case 2:
                      // easy, just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = from_Real16_t[i].to_Real32_t();
                      }
                      break;
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                      if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                        || (state == Signal::Stokes) ) {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(npol)
                              * uint_fast64_t(ndim);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                              into[i] = from_Real16_t[i].to_Real32_t();
                          }
                      }
                      else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(4);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                              into[i+0] = from_Real16_t[i+0].to_Real32_t();
                              into[i+1] = from_Real16_t[i+3].to_Real32_t();
                              into[i+2] = from_Real16_t[i+1].to_Real32_t();
                              into[i+3] = from_Real16_t[i+2].to_Real32_t();
                          }
                      }
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unrecognized pol_ordering and state combination");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  } // switch npol
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 16 bits");
              } // switch binary_format
              break;
          case 32: // nbit == 32
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 32 bits");
              }
              from_float = reinterpret_cast<const float* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  // easy, just copy over directly
                  NUM_POINTS = uint_fast64_t(ndat)
                      * uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                      into[i] = float(from_float[i]);
                  }
                  break;
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                    || (state == Signal::Stokes) ) {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = float(from_float[i]);
                      }
                  }
                  else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(4);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                          into[i+0] = float(from_float[i+0]);
                          into[i+1] = float(from_float[i+3]);
                          into[i+2] = float(from_float[i+1]);
                          into[i+3] = float(from_float[i+2]);
                      }
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized pol_ordering and state combination");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              } // switch npol
              break;
          case 64: // nbit == 64
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 64 bits");
              }
              from_double = reinterpret_cast<const double* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  // easy, just copy over directly
                  NUM_POINTS = uint_fast64_t(ndat)
                      * uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                      into[i] = float(from_double[i]);
                  }
                  break;
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                    || (state == Signal::Stokes) ) {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = float(from_double[i]);
                      }
                  }
                  else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(4);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                          into[i+0] = float(from_double[i+0]);
                          into[i+1] = float(from_double[i+3]);
                          into[i+2] = float(from_double[i+1]);
                          into[i+3] = float(from_double[i+2]);
                      }
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized pol_ordering and state combination");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              } // switch npol
              break;
          default:
              throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                           "unrecognized nbit");
          } // switch nbit
          break;
      }
      else { // NEED_TO_BYTESWAP
          into = reinterpret_cast<float* restrict>(output->get_dattfp());
          switch(nbit) {
          case 4: // nbit == 4
              // no actual need to byteswap here, but maintained for code simplicity
              if(ndim == 1) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "nbit == 4 and ndim == 1 not allowed for LuMP");
              }
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 4 bits");
              }
              from_uint8_t = reinterpret_cast<const uint8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  switch(ndim) {
                  case 2:
                      // easy, just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                                   * uint_fast64_t(nchan)
                                   * uint_fast64_t(npol)
                                   * uint_fast64_t(ndim) / 2;
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          // LOFAR sticks the real part into the lower
                          // 4 bits, and the imaginary part into the
                          // upper 4 bits.
                          uint8_t u = from_uint8_t[i];
                          *into++ = int4_t_values_Real32_t[u & 0xF];
                          *into++ = int4_t_values_Real32_t[u >> 4];
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unsupported ndim");
                  }
                  break;
              case 4:
                  if(ndim == 1) {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "nbit == 4 and ndim == 1 not allowed for LuMP");
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 8: // nbit == 8
              // no actual need to byteswap here, but maintained for code simplicity
              if(binary_format != dsp::IntegerBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 8 bits");
              }
              from_int8_t = reinterpret_cast<const int8_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  // easy, just copy over directly
                  NUM_POINTS = uint_fast64_t(ndat)
                      * uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                      into[i] = float(from_int8_t[i]);
                  }
                  break;
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                    || (state == Signal::Stokes) ) {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = float(from_int8_t[i]);
                      }
                  }
                  else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(4);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                          into[i+0] = float(from_int8_t[i+0]);
                          into[i+1] = float(from_int8_t[i+3]);
                          into[i+2] = float(from_int8_t[i+1]);
                          into[i+3] = float(from_int8_t[i+2]);
                      }
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized pol_ordering and state combination");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              }
              break;
          case 16: // nbit == 16
              // NEED_TO_BYTESWAP
              switch(binary_format) {
              case dsp::IntegerBinForm:
                  from_int16_t = reinterpret_cast<const int16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 1:
                  case 2:
                      // easy, just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = float(byteswap_int16_t(from_int16_t[i]));
                      }
                      break;
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                      if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                        || (state == Signal::Stokes) ) {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(npol)
                              * uint_fast64_t(ndim);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                              into[i] = float(byteswap_int16_t(from_int16_t[i]));
                          }
                      }
                      else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(4);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                              into[i+0] = float(byteswap_int16_t(from_int16_t[i+0]));
                              into[i+1] = float(byteswap_int16_t(from_int16_t[i+3]));
                              into[i+2] = float(byteswap_int16_t(from_int16_t[i+1]));
                              into[i+3] = float(byteswap_int16_t(from_int16_t[i+2]));
                          }
                      }
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unrecognized pol_ordering and state combination");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  }
                  break;
              case dsp::IEEE_FloatBinForm: // binary16 floating point format
                  from_uint16_t = reinterpret_cast<const uint16_t* restrict>(input->get_rawptr());
                  switch(npol) {
                  case 1:
                  case 2:
                      // easy, just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          X0.load_bits_byteswap(from_uint16_t[i]);
                          into[i] = X0.to_Real32_t();
                      }
                      break;
                  case 4:
                      if(ndim == 1) {}
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "npol == 4 and ndim != 1 not allowed for DSPSR");
                      }
                      if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                        || (state == Signal::Stokes) ) {
                          // DSPSR format is the same pol ordering as the LuMP data
                          // just copy over directly
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(npol)
                              * uint_fast64_t(ndim);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                              X0.load_bits_byteswap(from_uint16_t[i]);
                              into[i] = X0.to_Real32_t();
                          }
                      }
                      else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                          // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                          // PP PQ QP QQ, so we have to reorder the polarizations.
                          // This only works for ndim == 1
                          NUM_POINTS = uint_fast64_t(ndat)
                              * uint_fast64_t(nchan)
                              * uint_fast64_t(4);
                          for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                              X0.load_bits_byteswap(from_uint16_t[i+0]);
                              X1.load_bits_byteswap(from_uint16_t[i+1]);
                              X2.load_bits_byteswap(from_uint16_t[i+2]);
                              X3.load_bits_byteswap(from_uint16_t[i+3]);
                              into[i+0] = X0.to_Real32_t();
                              into[i+1] = X3.to_Real32_t();
                              into[i+2] = X1.to_Real32_t();
                              into[i+3] = X2.to_Real32_t();
                          }
                      }
                      else {
                          throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                       "unrecognized pol_ordering and state combination");
                      }
                      break;
                  default:
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized npol");
                  } // switch npol
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 16 bits");
              } // switch binary_format
              break;
          case 32: // nbit == 32
              // NEED_TO_BYTESWAP
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 32 bits");
              }
              from_int32_t = reinterpret_cast<const int32_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  // easy, just copy over directly
                  NUM_POINTS = uint_fast64_t(ndat)
                      * uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                      into[i] = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[i]));
                  }
                  break;
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                    || (state == Signal::Stokes) ) {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[i]));
                      }
                  }
                  else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(4);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                          into[i+0] = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[i+0]));
                          into[i+1] = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[i+3]));
                          into[i+2] = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[i+1]));
                          into[i+3] = int32_t_bits_to_float(byteswap_int32_t(from_int32_t[i+2]));
                      }
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized pol_ordering and state combination");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              } // switch npol
              break;
          case 64: // nbit == 64
              // NEED_TO_BYTESWAP
              if(binary_format != dsp::IEEE_FloatBinForm) {
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "invalid BinaryFormat for 64 bits");
              }
              from_int64_t = reinterpret_cast<const int64_t* restrict>(input->get_rawptr());
              switch(npol) {
              case 1:
              case 2:
                  // easy, just copy over directly
                  NUM_POINTS = uint_fast64_t(ndat)
                      * uint_fast64_t(nchan)
                      * uint_fast64_t(npol)
                      * uint_fast64_t(ndim);
                  for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                      into[i] = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[i]));
                  }
                  break;
              case 4:
                  if(ndim == 1) {}
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "npol == 4 and ndim != 1 not allowed for DSPSR");
                  }
                  if( ((state == Signal::Coherence) && (pol_ordering == dsp::DSPSROrdering))
                    || (state == Signal::Stokes) ) {
                      // DSPSR format is the same pol ordering as the LuMP data
                      // just copy over directly
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(npol)
                          * uint_fast64_t(ndim);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i++) {
                          into[i] = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[i]));
                      }
                  }
                  else if((state == Signal::Coherence) && (pol_ordering == dsp::JonesMatrixOrdering)) {
                      // DSPSR format is PP QQ Re[PQ] Im[PQ], but our format is
                      // PP PQ QP QQ, so we have to reorder the polarizations.
                      // This only works for ndim == 1
                      NUM_POINTS = uint_fast64_t(ndat)
                          * uint_fast64_t(nchan)
                          * uint_fast64_t(4);
                      for(uint_fast64_t i=0; i < NUM_POINTS; i+=4) {
                          into[i+0] = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[i+0]));
                          into[i+1] = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[i+3]));
                          into[i+2] = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[i+1]));
                          into[i+3] = int64_t_bits_to_double(byteswap_int64_t(from_int64_t[i+2]));
                      }
                  }
                  else {
                      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                                   "unrecognized pol_ordering and state combination");
                  }
                  break;
              default:
                  throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                               "unrecognized npol");
              } // switch npol
              break;
          default:
              throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                           "unrecognized nbit");
          } // switch nbit
          break;
      } // NEED_TO_BYTESWAP == true
      break;
  default:
      throw Error (InvalidState, "dsp::LuMPUnpacker::unpack",
                   "unrecognized order");
  } // switch output->get_order()
}

