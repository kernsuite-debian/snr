
INSTALL_ROOT ?= $(HOME)
INCLUDES := -I"include" -I"$(INSTALL_ROOT)/include"
LIBS := -L"$(INSTALL_ROOT)/lib"

CC := g++
CFLAGS := -std=c++11 -Wall
LDFLAGS := -lm -lOpenCL -lutils -lisaOpenCL -lAstroData

ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -O3 -g0
endif

ifdef PSRDADA
	CFLAGS += -DHAVE_PSRDADA
	LDFLAGS += -lpsrdada -lcudart
endif

all: bin/SNR.o bin/SNRTest bin/SNRTuning
	-@mkdir -p lib
	$(CC) -o lib/libSNR.so -shared -Wl,-soname,libSNR.so bin/SNR.o $(CFLAGS)

bin/SNR.o: include/SNR.hpp src/SNR.cpp
	-@mkdir -p bin
	$(CC) -o bin/SNR.o -c -fpic src/SNR.cpp $(INCLUDES) $(CFLAGS)

bin/SNRTest: src/SNRTest.cpp
	-@mkdir -p bin
	$(CC) -o bin/SNRTest src/SNRTest.cpp bin/SNR.o $(INCLUDES) $(LIBS) $(LDFLAGS) $(CFLAGS)

bin/SNRTuning: src/SNRTuning.cpp
	-@mkdir -p bin
	$(CC) -o bin/SNRTuning src/SNRTuning.cpp bin/SNR.o $(INCLUDES) $(LIBS) $(LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*
	-@rm lib/*

install: all
	-@mkdir -p $(INSTALL_ROOT)/include
	-@cp include/SNR.hpp $(INSTALL_ROOT)/include
	-@mkdir -p $(INSTALL_ROOT)/lib
	-@cp lib/* $(INSTALL_ROOT)/lib
	-@mkdir -p $(INSTALL_ROOT)/bin
	-@cp bin/SNRTest $(INSTALL_ROOT)/bin
	-@cp bin/SNRTuning $(INSTALL_ROOT)/bin
