dnl @synopsis SWIN_LIB_MKL
dnl 
AC_DEFUN([SWIN_LIB_MKL],
[
  AC_PROVIDE([SWIN_LIB_MKL])

  AC_ARG_WITH([mkl-dir],
              AC_HELP_STRING([--with-mkl-dir=DIR],
                             [Intel Math Kernel Library is in DIR]))

  MKL_LIBS=""
  MKL_CFLAGS=""

  if test x"$with_mkl_dir" = xno; then
    # user disabled mkl. Leave cache alone.
    have_mkl="User disabled MKL."
  else

    # "yes" is not a specification
    if test x"$with_mkl_dir" = xyes; then
      with_mkl_dir=
    fi

    AC_MSG_CHECKING([for Intel Math Kernel Library (MKL) installation])

    ## Look for the library ##
    cf_lib_path_list="$with_mkl_lib_dir .
                      /usr/local/intel/mkl72/lib/32
                      /usr/local/intel/mkl61/lib/32
                      /opt/intel/mkl61/lib/32
		      /import/psr/intel/mkl61/lib/32
		      /import/cluster/packages/linux/lib
		      /import/psrspin1/ord/packages/intel/mkl721/lib/32"	

    ac_save_LIBS="$LIBS"
    ac_test_LIBS="-lmkl_p4 -lguide -lpthread -lm"

    for cf_dir in $cf_lib_path_list; do
      LIBS="-L$cf_dir $ac_test_LIBS $ac_save_LIBS"
      AC_TRY_LINK([void cfft1d_(float*,int*,int*,float*);],
                  [cfft1d_(0,0,0,0);],
                  have_mkl=yes, have_mkl=no)
      if test x"$have_mkl" = xyes; then
        if test x"$cf_dir" = x.; then
          MKL_LIBS=$ac_test_LIBS
        else
          MKL_LIBS="-L$cf_dir $ac_test_LIBS"
        fi
        break
      fi
    done

    LIBS="$ac_save_LIBS"

  fi

  AC_MSG_RESULT([$have_mkl])

  if test x"$have_mkl" = xyes; then
    AC_DEFINE([HAVE_MKL],[1],
              [Define if the Intel Math Kernel Library is present])
    [$1]
  else
    :
    [$2]
  fi

  AC_SUBST(MKL_LIBS)
  AC_SUBST(MKL_CFLAGS)
  AM_CONDITIONAL(HAVE_MKL,[test x"$have_mkl" = xyes])

])

