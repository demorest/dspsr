
bin_PROGRAMS = load_bits

check_PROGRAMS = test_Input test_Unpack

load_bits_SOURCES = load_bits.C
test_Input_SOURCES = test_Input.C
test_Unpack_SOURCES = test_Unpack.C

if HAVE_sigproc
  check_PROGRAMS += sigproc_header
  sigproc_header_SOURCES = sigproc_header.C
endif

#############################################################################

INCLUDES = -I$(top_builddir)/local_include 

LDADD = $(top_builddir)/Kernel/libdspbase.la @CUDA_LIBS@ @CUFFT_LIBS@

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @CUDA_CFLAGS@ @CUFFT_CFLAGS@

