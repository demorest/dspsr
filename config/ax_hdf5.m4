dnl
dnl NOTE: this file has been modified from its original form on 9/22/2015.
dnl
dnl ######################################################################
dnl
dnl File:	hdf5.m4
dnl
dnl Purpose:	Determine the locations of hdf5 includes and libraries
dnl
dnl Version: $Id: hdf5.m4,v 1.26 2003/09/15 20:36:26 cary Exp $
dnl
dnl Tech-X configure system
dnl
dnl Copyright Tech-X Corporation
dnl
dnl ######################################################################
dnl

dnl
dnl NOTE: this file was retrieved from:
dnl
dnl   https://www.hdfgroup.org/ftp/HDF5/contrib/autoconf-macros/hdf5.m4
dnl

dnl
dnl Copyright Notice and License Terms for 
dnl HDF5 (Hierarchical Data Format 5) Software Library and Utilities
dnl -----------------------------------------------------------------------------
dnl 
dnl HDF5 (Hierarchical Data Format 5) Software Library and Utilities
dnl Copyright 2006-2015 by The HDF Group.
dnl 
dnl NCSA HDF5 (Hierarchical Data Format 5) Software Library and Utilities
dnl Copyright 1998-2006 by the Board of Trustees of the University of Illinois.
dnl 
dnl All rights reserved.
dnl 
dnl Redistribution and use in source and binary forms, with or without 
dnl modification, are permitted for any purpose (including commercial purposes) 
dnl provided that the following conditions are met:
dnl 
dnl 1. Redistributions of source code must retain the above copyright notice, 
dnl    this list of conditions, and the following disclaimer.
dnl 
dnl 2. Redistributions in binary form must reproduce the above copyright notice, 
dnl    this list of conditions, and the following disclaimer in the documentation 
dnl    and/or materials provided with the distribution.
dnl 
dnl 3. In addition, redistributions of modified forms of the source or binary 
dnl    code must carry prominent notices stating that the original code was 
dnl    changed and the date of the change.
dnl 
dnl 4. All publications or advertising materials mentioning features or use of 
dnl    this software are asked, but not required, to acknowledge that it was 
dnl    developed by The HDF Group and by the National Center for Supercomputing 
dnl    Applications at the University of Illinois at Urbana-Champaign and 
dnl    credit the contributors.
dnl 
dnl 5. Neither the name of The HDF Group, the name of the University, nor the 
dnl    name of any Contributor may be used to endorse or promote products derived 
dnl    from this software without specific prior written permission from 
dnl    The HDF Group, the University, or the Contributor, respectively.
dnl
dnl DISCLAIMER: 
dnl THIS SOFTWARE IS PROVIDED BY THE HDF GROUP AND THE CONTRIBUTORS 
dnl "AS IS" WITH NO WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED.  In no 
dnl event shall The HDF Group or the Contributors be liable for any damages 
dnl suffered by the users arising out of the use of this software, even if 
dnl advised of the possibility of such damage. 

AC_DEFUN([AX_HDF5], [

dnl ######################################################################
dnl
dnl Allow the user to specify an overall hdf5 directory.  If specified,
dnl we look for include and lib under this.
dnl
dnl ######################################################################

AC_ARG_WITH(hdf5,[  --with-hdf5=<location of hdf5 installation> ],HDF5_DIR="$withval",HDF5_DIR="")

dnl ######################################################################
dnl
dnl Find hdf5 includes - looking in include location if present,
dnl otherwise in dir/include if present, otherwise in default locations,
dnl first parallel, then serial.
dnl
dnl ######################################################################

AC_ARG_WITH(hdf5-incdir,[  --with-hdf5-incdir=<location of hdf5 includes> ],
HDF5_INCDIR="$withval",HDF5_INCDIR="")
if test "x$HDF5_DIR" != xno; then
if test -n "$HDF5_INCDIR"; then
  HDF5_INCPATH=$HDF5_INCDIR
elif test -n "$HDF5_DIR"; then
  HDF5_INCPATH=$HDF5_DIR/include
elif test "$MPI" = yes; then
  HDF5_INCPATH=$HOME/hdf5mpi/include:/usr/local/hdf5mpi/include:/loc/hdf5mpi/include:$HOME/hdf5/include:/usr/local/hdf5/include:/loc/hdf5/include:/usr/common/usg/hdf5/default/parallel/include:/usr/local/include
else
  HDF5_INCPATH=$HOME/hdf5/include:/usr/local/hdf5/include:/loc/hdf5/include:$HOME/hdf5mpi/include:/usr/local/hdf5mpi/include:/loc/hdf5mpi/include:/usr/common/usg/hdf5/default/serial/include
fi
saveCPPFLAGS=$CPPFLAGS
CPPFLAGS="-I$HDF5_INCPATH $CPPFLAGS"
AC_CHECK_HEADER(hdf5.h, [HDF5_H=y], [HDF5_H=""])
CPPFLAGS=$saveCPPFLAGS
if test -z "$HDF5_H"; then
  AC_MSG_WARN(hdf5.h not found in $HDF5_INCPATH.  Set with --with-hdf5-incdir=)
  HDF5_INC=" "
  ac_cv_have_hdf5=no
else
  HDF5_INCDIR=$HDF5_INCPATH
  AC_SUBST(HDF5_INCDIR)
  HDF5_INC=-I$HDF5_INCDIR
  HDF5_CPPFLAGS=$HDF5_INC
  AC_SUBST(HDF5_INC)
  AC_SUBST(HDF5_CPPFLAGS)
  HDF5_DIR=`dirname $HDF5_INCDIR`
  ac_cv_have_hdf5=yes
fi
fi
dnl ######################################################################
dnl
dnl See if built parallel
dnl
dnl ######################################################################

if test $ac_cv_have_hdf5 = yes; then
  if test -f $HDF5_INCDIR/H5config.h; then
    hdf5par=`grep "HAVE_PARALLEL 1" $HDF5_INCDIR/H5config.h`
  elif test -f $HDF5_INCDIR/H5pubconf.h; then
    hdf5par=`grep "HAVE_PARALLEL 1" $HDF5_INCDIR/H5pubconf.h`
  fi
fi

dnl ######################################################################
dnl
dnl Find hdf5 libraries
dnl
dnl ######################################################################

AC_ARG_WITH(hdf5-libdir,[  --with-hdf5-libdir=<location of hdf5 library> ],
HDF5_LIBDIR="$withval",HDF5_LIBDIR="")
if test $ac_cv_have_hdf5 = yes; then
  if test -n "$HDF5_LIBDIR"; then
    HDF5_LIBPATH=$HDF5_LIBDIR
  else
    HDF5_LIBPATH=$HDF5_DIR/lib
  fi
  
  saveLDFLAGS=$LDFLAGS
  LDFLAGS="-L$HDF5_LIBPATH $LDFLAGS"
  AC_CHECK_LIB([hdf5],[H5open],[LIBHDF5_A=y],[LIBHDF5_A=""])
  LDFLAGS=$saveLDFLAGS
  
  if test -z "$LIBHDF5_A"; then
    AC_MSG_WARN(libhdf5.a not found.  Set with --with-hdf5-libdir=)
    ac_cv_have_hdf5=no
    HDF5_LDFLAGS=" "
    HDF5_LIBS=" "
  else
    HDF5_LIBDIR=$HDF5_LIBPATH
    AC_SUBST(HDF5_LIBDIR)
    HDF5_LDFLAGS="-L$HDF5_LIBDIR"
    HDF5_LIBS="-lhdf5"
    AC_SUBST(HDF5_LDFLAGS)
    AC_SUBST(HDF5_LIBS)
  fi
fi

dnl ######################################################################
dnl
dnl Define for whether hdf5 found
dnl
dnl ######################################################################

if test $ac_cv_have_hdf5 = yes; then
  AC_DEFINE(HAVE_HDF5, [1], [Define if we have libhdf5])
  AM_CONDITIONAL(HAVE_HDF5, true)
else
  AM_CONDITIONAL(HAVE_HDF5, false)
fi

]) dnl End of DEFUN
