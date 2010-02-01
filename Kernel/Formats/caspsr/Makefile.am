
noinst_LTLIBRARIES = libcaspsr.la

nobase_include_HEADERS =  dsp/CASPSRUnpacker.h

libcaspsr_la_SOURCES = CASPSRUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/CASPSRUnpackerCUDA.h
libcaspsr_la_SOURCES += CASPSRUnpackerCUDA.cu

libcaspsr_la_LIBADD = CASPSRUnpackerCUDA.lo 

CASPSRUnpackerCUDA.lo : CASPSRUnpackerCUDA.cu
	@CUDA_NVCC@ -c $(CPPFLAGS) @PSRCHIVE_CFLAGS@ CASPSRUnpackerCUDA.cu
	sed -e 's/CASPSRUnpacker/CASPSRUnpackerCUDA/' CASPSRUnpacker.lo > CASPSRUnpackerCUDA.lo 

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @CUDA_CFLAGS@
