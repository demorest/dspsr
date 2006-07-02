
SUBDIRS	= base backends dsp psr

libs: install-binSCRIPTS
	$(MAKE) RECURSIVE_TARGETS="all-recursive libs-recursive" libs-recursive

all: config

config := $(srcdir)/config/Makefile.local

config : $(config)

#
# Convert automake variables into Standard Makefile Scheme variables
#
$(config): $(srcdir)/Makefile.am
	@echo "CC := $(CC)" > $@
	@echo "CCC := $(CXX)" >> $@
	@echo "FC := $(F77)"  >> $@
	@echo "LF77 := @FLIBS@" >> $@
	@echo "F77LIBS := $$(LF77)" >> $@
	$(pgplot.config)

if HAVE_PGPLOT
define pgplot.config 
@echo "has_PGPLOT := yes" >> $@
@echo "LCPGPLOT := @PGPLOT_LIBS@" >> $@
@echo "CFLAGS += @PGPLOT_CFLAGS@" >> $@
endef
endif

backends: backends.list

##############################################################################
#
# Rule to patch Makefile.local with new entries in Makefile.local_template
#
backends.list : available.backends
	@if [ ! -f $@ ]; then cp $< $@; fi;
	@diff $@ $< | grep -A1 -E '^[0-9]+a[0-9]+$$' | patch $@
	@touch $@
	@echo "************************************************************"
	@echo "              backends.list has been updated"
	@echo "************************************************************"

