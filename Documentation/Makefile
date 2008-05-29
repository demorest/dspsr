
default:
	@cp `psrchive_ssi` html || echo PSRCHIVE tags not available
	doxygen dspsr.dox > doxygen.out 2> doxygen.err
	@grep '^Error' doxygen.err || echo -n

install:
	./upload.csh

