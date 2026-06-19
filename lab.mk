ARCH      ?= native
LDFLAGS   ?=
INCLUDES  ?=
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) $(INCLUDES)
CXXFLAGS  := -O3 -std=c++17 -march=native

CU_BINS  := $(patsubst %.cu,%,$(wildcard *.cu))
CPP_BINS := $(patsubst %.cpp,%,$(wildcard *.cpp))

all: out $(CU_BINS) $(CPP_BINS)
out: ; mkdir -p out
%: %.cu  ; nvcc $(NVCCFLAGS) $(LDFLAGS) $< -o $@
%: %.cpp ; g++  $(CXXFLAGS) $< -o $@
clean:   ; rm -f $(CU_BINS) $(CPP_BINS)