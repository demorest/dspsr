dnl @synopsis SWIN_LIB_CUFFT
dnl 
AC_DEFUN([SWIN_LIB_CUFFT],
[
  AC_PROVIDE([SWIN_LIB_CUFFT])

  AC_ARG_WITH([cufft],
              AC_HELP_STRING([--with-cufft],
                             [Compile CUFFT-based filterbank engine]))

  CUFFT_CFLAGS=""
  CUFFT_LIBS=""

  if test x"$with_cufft" = x"yes"; then

    AC_MSG_NOTICE([CUFFT enabled.])

    AC_DEFINE([HAVE_CUFFT],[1],
              [Define if CUFFT library is enabled])

    # Swinburne-specific for now
    CUFFT_LIBS="-L/opt/local/cuda/lib64 -lcufft"
    CUFFT_CFLAGS="-I/lfs/data0/bbarsdel/g2x/NVIDIA_CUDA_SDK/common/inc -I/opt/local/cuda/include"
  fi

  AC_SUBST(CUFFT_LIBS)
  AC_SUBST(CUFFT_CFLAGS)
  AM_CONDITIONAL(HAVE_CUFFT,[test x"$have_cufft" != xno])

])

