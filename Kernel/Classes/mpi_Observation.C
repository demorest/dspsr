#include "Observation.h"
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
  total_size += temp_size;  // nbit

  mpiPack_size (start_time, comm, &temp_size);
  total_size += temp_size;

  MPI_Pack_size (1, MPI_CHAR, comm, &temp_size);
  total_size += temp_size;  // state
  total_size += temp_size;  // feedtype
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
  int64 ndat = obs.get_ndat();
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

  obs.get_position().getRadians (&temp, &temp2);
  MPI_Pack (&temp, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);
  MPI_Pack (&temp2, 1, MPI_DOUBLE, outbuf, outcount, pos, comm);

  int tempi;
  tempi = obs.get_nchan();
  MPI_Pack (&tempi, 1, MPI_INT, outbuf, outcount, pos, comm);

  tempi = obs.get_npol();
  MPI_Pack (&tempi, 1, MPI_INT, outbuf, outcount, pos, comm);

  tempi = obs.get_nbit();
  MPI_Pack (&tempi, 1, MPI_INT, outbuf, outcount, pos, comm);

  // use mpiPack (MJD)

  mpiPack (start_time, outbuf, outcount, pos, comm);

  char tempc;
  tempc = obs.get_state();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_feedtype();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_swap();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_dc_centred();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  tempc = obs.get_telescope();
  MPI_Pack (&tempc, 1, MPI_CHAR, outbuf, outcount, pos, comm);

  // use mpiPack (string)

  mpiPack (source, outbuf, outcount, pos, comm);
  mpiPack (identifier, outbuf, outcount, pos, comm);
  mpiPack (mode, outbuf, outcount, pos, comm);
  mpiPack (machine, outbuf, outcount, pos, comm);

  return 0;
}

int observation::MpiUnpack (void* inbuf, int insize, int* pos, 
			    MPI_Comm comm)
{
  int64 ndat;

  MPI_Unpack (inbuf, insize, pos, &ndat, 1, MPI_Int64,  comm);
  obs.set_ndat (ndat);

  double temp, temp2;

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs.set_centre_frequency (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs.set_bandwidth (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs.set_rate (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  obs.set_scale (temp);

  MPI_Unpack (inbuf, insize, pos, &temp, 1, MPI_DOUBLE, comm);
  MPI_Unpack (inbuf, insize, pos, &temp2, 1, MPI_DOUBLE, comm);
  sky_coord position;
  position.setRadians(temp, temp2);
  obs.set_position (position);

  int tempi;

  MPI_Unpack (inbuf, insize, pos, &tempi, 1, MPI_INT, comm);
  obs.set_nchan (tempi);

  MPI_Unpack (inbuf, insize, pos, &tempi, 1, MPI_INT, comm);
  obs.set_npol (tempi);

  MPI_Unpack (inbuf, insize, pos, &tempi, 1, MPI_INT, comm);
  obs.set_nbit (tempi);

  // use mpiUnpack (MJD)
  mpiUnpack (inbuf, insize, pos, &start_time, comm);

  char tempc;
  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs.set_state (tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs.set_feedtype (tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs.set_swap (tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs.set_dc_centred (tempc);

  MPI_Unpack (inbuf, insize, pos, &tempc, 1, MPI_CHAR, comm);
  obs.set_telescope (tempc);

  // use mpiUnpack (string)

  mpiUnpack (inbuf, insize, pos, &pulsar, comm);
  mpiUnpack (inbuf, insize, pos, &identifier, comm);
  mpiUnpack (inbuf, insize, pos, &mode, comm);
  mpiUnpack (inbuf, insize, pos, &machine, comm);

  return 0;
}
