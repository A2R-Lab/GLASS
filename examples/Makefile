# Build all GLASS examples: `make [-j] [ARCH=sm_120] [MATHDX_ROOT=...]`
# ARCH auto-detects from nvidia-smi when unset; falls back to sm_90.
# Example 06 (glass::nvidia:: GEMM) builds only when MATHDX_ROOT is set.

NVCC      ?= nvcc
ARCH      ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d '.' | sed 's/^/sm_/' | grep -E 'sm_[0-9]+' || echo sm_90)
NVCCFLAGS ?= -std=c++17 -arch=$(ARCH) -I..

SRCS   := $(filter-out 05_nvidia_gemm.cu,$(wildcard [0-9][0-9]_*.cu))
BINS   := $(SRCS:.cu=)

ifdef MATHDX_ROOT
BINS += 05_nvidia_gemm
SMS  ?= $(subst sm_,,$(ARCH))0
endif

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

05_nvidia_gemm: 05_nvidia_gemm.cu
	$(NVCC) $(NVCCFLAGS) -I$(MATHDX_ROOT)/include -I$(MATHDX_ROOT)/external/cutlass/include -DGLASS_BENCH_CUBLASDX -DSMS=$(SMS) \
	    -Xptxas -O1 $< -o $@

run: all
	@for b in $(BINS); do echo "── $$b ──"; ./$$b || exit 1; done

clean:
	rm -f $(BINS)

.PHONY: all run clean
