
noinst_LTLIBRARIES = libvdif.la

nobase_include_HEADERS = dsp/VDIFFile.h dsp/VDIFTwoBitCorrection.h	\
			 dsp/VDIFTwoBitCorrectionMulti.h		\
			 dsp/VDIFTwoBitTable.h				\
			 dsp/VDIFFourBitUnpacker.h			\
			 dsp/VDIFEightBitUnpacker.h			\
			 dsp/VDIFnByteUnpacker.h

libvdif_la_SOURCES = VDIFFile.C \
		     VDIFTwoBitCorrection.C \
		     VDIFTwoBitCorrectionMulti.C \
		     VDIFTwoBitTable.C \
		     VDIFFourBitUnpacker.C \
		     VDIFEightBitUnpacker.C \
		     VDIFnByteUnpacker.C \
		     vdifio.c vdifio.h

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

