#include "CPSR2File.h"
#include "CPSRFile.h"
#include "S2File.h"
#include "PMDAQFile.h"
#include "DigiFile.h"

Registry::List<dsp::File> dsp::File::registry;

static Registry::List<dsp::File>::Enter<dsp::CPSR2File> cpsr2;
static Registry::List<dsp::File>::Enter<dsp::CPSRFile>  cpsr;
static Registry::List<dsp::File>::Enter<dsp::PMDAQFile> pmdaq;
static Registry::List<dsp::File>::Enter<dsp::S2File>    s2;

static Registry::List<dsp::File>::Enter<dsp::DigiFile> digifile;
