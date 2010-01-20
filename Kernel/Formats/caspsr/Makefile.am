
noinst_LTLIBRARIES = libcaspsr.la

nobase_include_HEADERS =  dsp/CASPSRUnpacker.h

libcaspsr_la_SOURCES = CASPSRUnpacker.C

if HAVE_CUFFT

nobase_include_HEADERS += dsp/CASPSRUnpackerCUDA.h dsp/CASPSRUnpackerSetup.h
libcaspsr_la_SOURCES += CASPSRUnpackerCUDA.cu CASPSRUnpackerSetup.C

libcaspsr_la_LIBADD = CASPSRUnpackerCUDA.lo 

CASPSRUnpackerCUDA.lo : CASPSRUnpackerCUDA.cu
	/usr/local/cuda/bin/nvcc -c -I/Developer/GPU\ Computing/C/common/inc -I../../local_include `psrchive --cflags` CASPSRUnpackerCUDA.cu
	sed -e 's/CASPSRUnpacker/CASPSRUnpackerCUDA/' CASPSRUnpacker.lo > CASPSRUnpackerCUDA.lo 

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

