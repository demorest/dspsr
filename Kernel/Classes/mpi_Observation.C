/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#define ACTIVATE_MPI 1
#include "dsp/Observation.h"
#include "stdmpi.h"
#include "portable_mpi.h"

int mpiPack_size (const dsp::Observation& obs, MPI_Comm comm, int* size)
{
  int total_size = 0;
  int temp_size = 0;

  MPI_Pack_size (1, MPI_Int64, comm, &temp_size); 
  total_size += temp_size;  // ndat

  MPI_Pack_size (1, MPI_DOUBLE, comm, &temp_size); 
  total_size += temp_size;  // centre_freq
  total_size += temp_size;  // bw
  total_size += temp_size;  // rate
  total_size += temp_size;  // scale
  total_size += temp_size;  // ra
  total_size += temp_size;  // dec

  MPI_Pack_size (1, MPI_INT, comm, &temp_size);
  total_size += temp_size;  // nchan
  total_size += temp_size;  // npol
  total_size += temp_size;  // ndim
  total_size += temp_size;  // nbit

  mpiPack_size (obs.get_start_time(), comm, &temp_size);
  total_size += temp_size;

  MPI_Pack_size (1, MPI_CHAR, comm, &temp_size);
  total_size += temp_size;  // state
  total_size += temp_size;  // basis
  total_size += temp_size;  // swap
  total_size += temp_size;  // dc_centred
  total_size += temp_size;  // telescope

  mpiPack_size  (obs.get_source(), comm, &temp_size);
  total_size += temp_size;
  mpiPack_size  (obs.get_identifier(), comm, &temp_size);
  total_size += temp_size;
  mpiPack_size  (obs.get_mode(), comm, &temp_size);
  total_size += temp_size;
  mpiPack_size  (obs.get_machine(), comm, &temp_size);
  total_size += temp_size;

  *size = total_size;
  return 1;  // no error, dynamic
}

int mpiPack (const dsp::Observation& obs,
	     void* outbuf, int outcount, int* pos, MPI_Comm comm)
{
  int64_t ndat = obs.get_ndat();
  MPI_Pack (&ndat, 1, MPI_Int64, outbuf, outcount, pos, comm);

  double temp, temp2;
  temp = obs.get_centre_frequency();
  MPI_Pack (&temp, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);

  temp = obs.get_bandwidth();
  MPI_Pack (&temp, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);

  temp = obs.get_rate();
  MPI_Pack (&temp, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);

  temp = obs.get_scale();
  MPI_Pack (&temp, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);

  obs.get_coordinates().getRadians (&temp, &temp2);
  MPI_Pack (&temp, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);
  MPI_Pack (&temp2, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);

  int tempi;
  tempi = obs.get_nchan();
  MPI_Pack (&tempi, 1, MPI_INT, outbuf, outcount, pos, comm);

  tempi = obs.get_npol();
  MPI_Pack (&tempi, 1, MPI_INT, outbuf, outcount, pos, comm);

  tempi = obs.get_ndim();
  MPI_Pack (&tempi, 1, MPI_INT, outbuf, outcount, pos, comm);

  tempi = obs.get_nbit();
  MPI_Pack (&tempi, 1, MPI_INT, outbuf, outcount, pos, comm);

  // use mpiPack (MJD)

  mpiPack (obs.get_start_time(), outbuf, outcount, pos, comm);

  char tempc;
  tempc = obs.get_state();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_basis();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_swap();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_dc_centred();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_telescope_code();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  // use mpiPack (string)

  mpiPack (obs.get_source(), outbuf, outcount, pos, comm);
  mpiPack (obs.get_identifier(), outbuf, outcount, pos, comm);
  mpiPack (obs.get_mode(), outbuf, outcount, pos, comm);
  mpiPack (obs.get_machine(), outbuf, outcount, pos, comm);

  return 0;
}

int mpiUnpack (void* inbuf, int insize, int* pos, dsp::Observation* obs,
               MPI_Comm comm)
{
  int64_t ndat;

  MPI_Unpack (inbuf, insize, pos, &ndat, 1, MPI_Int64,  comm);
  obs->set_ndat (ndat);

  double temp, temp2;

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs->set_centre_frequency (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs->set_bandwidth (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs->set_rate (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs->set_scale (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  MPI_Unpack (inbuf, insize, pos, &temp2, 1, MPI_DOUBLE, comm);
  sky_coord coordinates;
  coordinates.setRadians(temp, temp2);
  obs->set_coordinates (coordinates);

  int tempi;

  MPI_Unpack (inbuf, insize, pos, &tempi, 1, MPI_INT, comm);
  obs->set_nchan (tempi);

  MPI_Unpack (inbuf, insize, pos, &tempi, 1, MPI_INT, comm);
  obs->set_npol (tempi);

  MPI_Unpack (inbuf, insize, pos, &tempi, 1, MPI_INT, comm);
  obs->set_ndim (tempi);

  MPI_Unpack (inbuf, insize, pos, &tempi, 1, MPI_INT, comm);
  obs->set_nbit (tempi);

  // use mpiUnpack (MJD)
  MJD tempmjd;
  mpiUnpack (inbuf, insize, pos, &tempmjd, comm);
  obs->set_start_time (tempmjd);

  char tempc;
  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs->set_state ((Signal::State) tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs->set_basis ((Signal::Basis) tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs->set_swap (tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs->set_dc_centred (tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs->set_telescope_code (tempc);

  // use mpiUnpack (string)

  string tempstr;
  mpiUnpack (inbuf, insize, pos, &tempstr, comm);
  obs->set_source (tempstr);

  mpiUnpack (inbuf, insize, pos, &tempstr, comm);
  obs->set_identifier (tempstr);

  mpiUnpack (inbuf, insize, pos, &tempstr, comm);
  obs->set_mode (tempstr);

  mpiUnpack (inbuf, insize, pos, &tempstr, comm);
  obs->set_machine (tempstr);

  return 0;
}
