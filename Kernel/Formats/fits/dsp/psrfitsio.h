//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/fits/dsp/Attic/psrfitsio.h,v $
   $Revision: 1.2 $
   $Date: 2009/04/15 06:35:04 $
   $Author: jonathan_khoo $ */

#ifndef __psrfitsio_h
#define __psrfitsio_h

#include "FITSError.h"
#include "fitsutil.h"

#include <fitsio.h>

#include <string>
#include <vector>

//! Empty template class requires specialization 
template<typename T> struct FITS_traits { };
  
//! Template specialization for double
template<> struct FITS_traits<double>
{
  static inline int datatype() { return TDOUBLE; }
  static inline double null () { return 0; }
};

//! Template specialization for float
template<> struct FITS_traits<float>
{
  static inline int datatype() { return TFLOAT; }
  static inline float null () { return fits_nullfloat; }
};

//! Template specialization for float
template<> struct FITS_traits<short>
{
  static inline int datatype() { return TSHORT; }
  static inline short null () { return -1; }
};

//! Template specialization for int
template<> struct FITS_traits<int>
{
  static inline int datatype() { return TINT; }
  static inline int null () { return -1; }
};

//! Template specialization for long
template<> struct FITS_traits<long> {
  static inline int datatype() { return TLONG; }
  static inline long null () { return -1; }
};

//! Template specialization for long
template<> struct FITS_traits<std::string>
{
  static inline int datatype() { return TSTRING; }
  static inline std::string null () { return ""; }
};

template<typename T>
void* FITS_void_ptr (const T& ptr)
{
  return const_cast<T*> (&ptr);
}

void* FITS_void_ptr (const std::string&);

//! Calls fits_update_key; throws a FITSError exception if status != 0
template<typename T>
void psrfits_update_key (fitsfile* fptr, const char* name, T data,
			 const char* comment = 0)
{
  // status
  int status = 0;
  
  fits_update_key (fptr, FITS_traits<T>::datatype(),
		   const_cast<char*>(name), &data,
		   const_cast<char*>(comment), &status);
  
  if (status)
    throw FITSError (status, "psrfits_update_key", name);
}

//! Specialization for C string
void psrfits_update_key (fitsfile* fptr, const char* name,
			 const char* data,
			 const char* comment = 0);

//! Specialization for C++ string
void psrfits_update_key (fitsfile* fptr, const char* name,
			 const std::string& data,
			 const char* comment = 0);

//! Convenience interface to fits_make_keyn + fits_update_key
void psrfits_update_key (fitsfile* fptr,
			 const char* name,
			 int column,
			 const std::string& data,
			 const char* comment = 0);

//! Worker function does not handle status
template<typename T>
void psrfits_read_key_work (fitsfile* fptr, const char* name, T* data, 
			    int* status)
{
  // no comment
  char* comment = 0;
  
  fits_read_key (fptr, FITS_traits<T>::datatype(), 
		 const_cast<char*>(name), data,
		 comment, status);
}

//! Specialization for string
void psrfits_read_key_work (fitsfile* fptr, const char* name, std::string*,
			    int* status);

//! Calls fits_read_key; throws a FITSError exception if status != 0
template<typename T>
void psrfits_read_key (fitsfile* fptr, const char* name, T* data)
{
  // status
  int status = 0;
  psrfits_read_key_work (fptr, name, data, &status);
  if (status)
    throw FITSError (status, "psrfits_read_key", name);
}

//! Calls fits_read_key; sets data to default if status != 0
template<typename T>
void psrfits_read_key (fitsfile* fptr, const char* name, T* data,
		       T dfault, bool verbose = false)
{
  // status
  int status = 0;
  psrfits_read_key_work (fptr, name, data, &status);
  if (status)
  {
    if (verbose)
    {
      FITSError error (status, "psrfits_read_key", name);
      std::cerr << error.get_message() << std::endl;
      std::cerr << "psrfits_read_key: using default=" << dfault << std::endl;
    }
    *data = dfault;
  }
  else if (verbose)
    std::cerr << "psrfits_read_key: " << name << "=" << *data << std::endl;
}

//! Write/update a 1-D TDIM descriptor for the specified column
void psrfits_update_tdim (fitsfile* ffptr, int column, unsigned dim);

//! Write/update a 2-D TDIM descriptor for the specified column
void psrfits_update_tdim (fitsfile* ffptr, int column,
			  unsigned dim1, unsigned dim2);

//! Write/update a 3-D TDIM descriptor for the specified column
void psrfits_update_tdim (fitsfile* ffptr, int column,
			  unsigned dim1, unsigned dim2, unsigned dim3);

//! Write/update a 4-D TDIM descriptor for the specified column
void psrfits_update_tdim (fitsfile* ffptr, int column,
			  unsigned dim1, unsigned dim2,
			  unsigned dim3, unsigned dim4);

//! Write/update a N-D TDIM descriptor for the specified column
void psrfits_update_tdim (fitsfile* ffptr, int column,
			  const std::vector<unsigned>& dims);

template<typename T>
void psrfits_write_col (fitsfile* fptr, const char* name, int row,
			const std::vector<T>& data,
			const std::vector<unsigned>& dims)
{
  //
  // Get the number of the named column

  int colnum = 0;
  int status = 0;

  fits_get_colnum (fptr, CASEINSEN, const_cast<char*>(name), &colnum, &status);

  fits_modify_vector_len (fptr, colnum, data.size(), &status);

  if (dims.size() > 1)
    psrfits_update_tdim (fptr, colnum, dims);

  fits_write_col (fptr, FITS_traits<T>::datatype(),
		  colnum, row,
		  1, data.size(),
		  const_cast<T*>(&(data[0])), &status);

  if (status)
    throw FITSError (status, "psrfits_write_col(vector<T>)", name);
}

template<typename T>
void psrfits_write_col( fitsfile *fptr, const char *name, int row,
			const T& data )
{
  int colnum = 0;
  int status = 0;
  
  fits_get_colnum (fptr, CASEINSEN, const_cast<char*>(name), &colnum, &status);
  
  fits_write_col (fptr, FITS_traits<T>::datatype(),
		  colnum, row,
		  1, 1,
		  FITS_void_ptr(data), &status);

  if (status)
    throw FITSError (status, "psrfits_write_col(T)", name);
}

template<typename T>
void psrfits_read_col (fitsfile* fptr, const char* name,
		       std::vector< std::vector<T> >& data,
		       int row = 1, T null = FITS_traits<T>::null())
{
  //
  // Get the number of the named column

  int colnum = 0;
  int status = 0;

  fits_get_colnum (fptr, CASEINSEN, const_cast<char*>(name), &colnum, &status);

  int anynul = 0;
  int counter = 1;

  for (unsigned i = 0; i < data.size(); i++)
  {
    fits_read_col (fptr, FITS_traits<T>::datatype(),
		   colnum, row, 
		   counter, data[i].size(),
		   &null, &(data[i][0]),
		   &anynul, &status);

    if (status != 0)
      throw FITSError( status, "psrfits_read_col",
		       "%s colnum=%d firstrow=%d firstelem=%d nelements=%d",
		       name, colnum, row, counter, data[i].size() );
      
    counter += data[i].size();
  }

}

template<typename T>
void psrfits_read_col (fitsfile* fptr, const char* name, std::vector<T>& data,
		       int row = 1, T null = FITS_traits<T>::null())
{
  //
  // Get the number of the named column

  int colnum = 0;
  int status = 0;

  fits_get_colnum (fptr, CASEINSEN, const_cast<char*>(name), &colnum, &status);

  //
  // If the vector length is unspecified, read it from the file

  if (data.size() == 0)
  {
    int typecode = 0;
    long repeat = 0;
    long width = 0;
  
    fits_get_coltype (fptr, colnum, &typecode, &repeat, &width, &status);  
    if (status)
      throw FITSError (status, "psrfits_read_col",
		       "fits_get_coltype (%s)", name);

    data.resize( repeat );
  }

  int anynul = 0;
  fits_read_col (fptr, FITS_traits<T>::datatype(),
		 colnum, row,
		 1, data.size(),
		 &null, &(data[0]),
		 &anynul, &status);

  if (status)
    throw FITSError (status, "psrfits_read_col", name);
}

//! Worker function does not handle status
template< typename T >
void psrfits_read_col_work( fitsfile *fptr, const char *name, T *data,
			    int row, T null, int* status)
{ 
  int colnum = 0;
  fits_get_colnum (fptr, CASEINSEN, const_cast<char*>(name), &colnum, status);
  
  int anynul = 0;
  fits_read_col( fptr, FITS_traits<T>::datatype(),
		 colnum, row,
		 1, 1, &null, data, 
		 &anynul, status );
}

//! Specialization for string
void psrfits_read_col_work( fitsfile *fptr, const char *name,
			    std::string *data,
			    int row, std::string& null, int* status);


template< typename T >
void psrfits_read_col( fitsfile *fptr, const char *name, T *data,
		       int row = 1, T null = FITS_traits<T>::null() )
{
  int status = 0;
  psrfits_read_col_work (fptr, name, data, row, null, &status);
  if (status)
    throw FITSError (status, "psrfits_read_call", name);
}

//! Calls fits_read_col; sets data to default if status != 0
template< typename T >
void psrfits_read_col( fitsfile *fptr, const char *name, T *data,
		       int row, T null, T dfault, bool verbose)
{
  // status
  int status = 0;
  psrfits_read_col_work (fptr, name, data, row, null, &status);
  if (status)
  {
    if (verbose)
    {
      FITSError error (status, "psrfits_read_col", name);
      std::cerr << error.get_message() << std::endl;
      std::cerr << "psrfits_read_col: using default=" << dfault << std::endl;
    }
    *data = dfault;
  }
  else if (verbose)
    std::cerr << "psrfits_read_col: " << name << "=" << *data << std::endl;

}

//! Move to the named HDU, remove any existing rows, and insert a new one
void psrfits_init_hdu (fitsfile *fptr, const char *name);

//! Move to the named HDU
void psrfits_move_hdu (fitsfile *fptr, const char *name,
		       int table_type = BINARY_TBL, int version = 0);

//! Remove any existing rows from the current binary table
void psrfits_clean_rows (fitsfile*);

//! Insert a single new row at the start of the current binary table
void psrfits_insert_row (fitsfile* fptr);

//! Delete the named column from the current binary table
void psrfits_delete_col (fitsfile* fptr, const char* name);

#endif
