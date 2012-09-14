# SWIN_LIB_CUDA([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
# ----------------------------------------------------------
AC_DEFUN([SWIN_LIB_CUDA],
[
  AC_PROVIDE([SWIN_LIB_CUDA])

  SWIN_PACKAGE_OPTIONS([cuda])

  CUDA_CFLAGS=""
  CUDA_LIBS=""

  if test x"$with_cuda_dir" = xno; then

    # user disabled cuda. Leave cache alone.
    have_cuda="User disabled CUDA."

  else

    # "yes" is not a specification
    if test x"$with_cuda_dir" = xyes; then
      with_cuda_dir=
    fi

    if test x"$with_cuda_dir" != x; then
      cuda_nvcc=$with_cuda_dir/bin/nvcc
    else
      AC_PATH_PROG(cuda_nvcc, nvcc, no)
    fi

    have_cuda="not found"

    AC_MSG_CHECKING([for CUDA installation])

    if test -x $cuda_nvcc; then

      CUDA_NVCC="$cuda_nvcc \$(CUDA_NVCC_FLAGS) \$(DEFAULT_INCLUDES) \$(INCLUDES) \$(AM_CPPFLAGS) \$(CPPFLAGS)"

      SWIN_PACKAGE_FIND([cuda],[cuda_runtime.h])
      SWIN_PACKAGE_TRY_COMPILE([cuda],[#include <cuda_runtime.h>])

      SWIN_PACKAGE_FIND([cuda],[libcudart.*])
      SWIN_PACKAGE_TRY_LINK([cuda],[#include <cuda_runtime.h>],
                            [cudaMalloc (0, 0);],[-lcudart])

    fi

  fi

  AC_MSG_RESULT([$have_cuda])

  if test "$have_cuda" = "yes"; then

    AC_DEFINE([HAVE_CUDA],[1],[Define if the CUDA library is present])

    AC_MSG_CHECKING([for CUDA FFT installation])

    SWIN_PACKAGE_FIND([cufft],[cufft.h])
    SWIN_PACKAGE_TRY_COMPILE([cufft],[#include <cufft.h>])

    SWIN_PACKAGE_FIND([cufft],[libcufft.*])
    SWIN_PACKAGE_TRY_LINK([cufft],[#include <cufft.h>],
                          [cufftPlan1d (0, 1024, CUFFT_C2C, 1);],[-lcufft])

    AC_MSG_RESULT([$have_cufft])

    if test "$have_cufft" = "yes"; then
      AC_DEFINE([HAVE_CUFFT],[1],[Define if the CUFFT library is present])
      [$1]
    else
      AC_MSG_WARN([dspsr will not run on GPU])
      [$2]
    fi

  else

    if test "$have_cuda" = "not found"; then
      echo
      AC_MSG_NOTICE([Ensure that CUDA nvcc is in PATH.])
      AC_MSG_NOTICE([Alternatively, use the --with-cuda-dir option.])
      echo
    fi

    [$2]

  fi

  AC_SUBST(CUDA_NVCC)

  CUDA_LIBS="$cuda_LIBS"
  CUDA_CFLAGS="$cuda_CFLAGS"

  AC_SUBST(CUDA_LIBS)
  AC_SUBST(CUDA_CFLAGS)
  AM_CONDITIONAL(HAVE_CUDA,[test "$have_cuda" = "yes"])

  CUFFT_LIBS="$cufft_LIBS"
  CUFFT_CFLAGS="$cufft_CFLAGS"

  AC_SUBST(CUFFT_LIBS)
  AC_SUBST(CUFFT_CFLAGS)
  AM_CONDITIONAL(HAVE_CUFFT,[test "$have_cufft" = "yes"])

])
