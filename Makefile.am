
SUBDIRS	= util base backends dsp psr

libs:
	$(MAKE) RECURSIVE_TARGETS="all-recursive libs-recursive" libs-recursive

all: $(config)

config := $(srcdir)/config/Makefile.local

#
# Convert automake variables into Standard Makefile Scheme variables
#
$(config): $(srcdir)/Makefile.am
	@echo "prefix := $(prefix)" > $@
	@echo "CC := $(CC)" >> $@
	@echo "CFLAGS := $(CFLAGS)" >> $@
	@echo "CCC := $(CXX)" >> $@
	@echo "CCFLAGS := $(CXXFLAGS)" >> $@
	@echo "FC := $(F77)"  >> $@
	@echo "FFLAGS := $(FFLAGS)" >> $@
	@echo "LF77 := @FLIBS@" >> $@
	@echo "LPSRCHIVE := @PSRCHIVE_LIBS@" >> $@
	@echo "INCFLAGS += @PSRCHIVE_CFLAGS@" >> $@
	$(pgplot.config)

if HAVE_PGPLOT
define pgplot.config 
@echo "has_PGPLOT := yes" >> $@
@echo "LCPGPLOT := @PGPLOT_LIBS@" >> $@
@echo "INCFLAGS += @PGPLOT_CFLAGS@" >> $@
endef
endif

