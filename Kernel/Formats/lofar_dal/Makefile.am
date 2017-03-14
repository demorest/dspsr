
noinst_LTLIBRARIES = liblofar_dal.la

nobase_include_HEADERS =  dsp/LOFAR_DALFile.h dsp/LOFAR_DALUnpacker.h

liblofar_dal_la_SOURCES = LOFAR_DALFile.C LOFAR_DALUnpacker.C 

liblofar_dal_la_LIBADD = -L$(LOFAR_DAL_INSTALL_PREFIX)/lib -llofardal @HDF5_LDFLAGS@ @HDF5_LIBS@

#############################################################################

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += -I$(LOFAR_DAL_INSTALL_PREFIX)/include/dal @HDF5_CPPFLAGS@


