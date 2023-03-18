.PHONY: help

# e.g., 
# time make all TARGET=hw PLATFORM=xilinx_u280_xdma_201920_3 > out 2>&1
# time make all TARGET=hw PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1 > out 2>&1

help::
        $(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make xclbin TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command compile just the kernel of the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make exe TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to generate the host application for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command compile just the kernel of the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""

# VPP := $(XILINX_VITIS)/bin/v++
VPP := v++
EMCONFIGUTIL := $(XILINX_VITIS)/bin/emconfigutil
TARGET := sw_emu
PLATFORM := xilinx_u250_gen3x16_xdma_4_1_202210_1

XCLBIN_DIR := ./xclbin
TEMP_DIR := ./_x.$(TARGET)
# sources
KERNEL_SRC := src/vadd.cpp
HOST_SRC := src/host.cpp

# targets
HOST_EXE := host

XOS := $(XCLBIN_DIR)/vadd.$(TARGET).xo
XCLBIN := $(XCLBIN_DIR)/vadd.$(TARGET).xclbin
EMCONFIG_FILE := ./emconfig.json

VPP_COMMON_OPTS := -g -t $(TARGET) --platform $(PLATFORM) --save-temps --config design.cfg
CFLAGS := -g -std=c++11 -I$(XILINX_XRT)/include
LFLAGS := -L$(XILINX_XRT)/lib -lxilinxopencl -pthread -lrt
NUMDEVICES := 1

# run time args
# EXE_OPT := vadd.$(TARGET).xclbin

# primary build targets
.PHONY: xclbin exe all

xclbin: $(XCLBIN)

exe: $(HOST_EXE)

all: exe xclbin $(EMCONFIG_FILE)



# kernel rules
$(XOS): $(KERNEL_SRC)
# $(RM) $@
	mkdir -p $(XCLBIN_DIR)
	$(VPP) $(VPP_COMMON_OPTS) -c -k vadd --temp_dir $(TEMP_DIR) -o $@ $+

$(XCLBIN): $(XOS)
	mkdir -p $(XCLBIN_DIR)
	$(VPP) $(VPP_COMMON_OPTS) -l --temp_dir $(TEMP_DIR) -o $@ $+ # $(VPP_LINK_OPTS)

# host rules
$(HOST_EXE): $(HOST_SRC)
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo 'Compiled Host Executable: $(HOST_EXE)'

$(EMCONFIG_FILE):
	$(EMCONFIGUTIL) --nd $(NUMDEVICES) --od . --platform $(PLATFORM)

run: $(XCLBIN) $(HOST_EXE) $(EMCONFIG_FILE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	export XCL_EMULATION_MODE=${TARGET}; ./$(HOST_EXE) $(XCLBIN)
else
	./$(HOST_EXE) $(XCLBIN)
endif


.PHONY: clean cleanall

clean:
	-$(RM) $(EMCONFIG_FILE) $(HOST_EXE) $(XCLBIN) *.xclbin *.xo $(XOS) *.log *.csv *summary *.json *.xml
	
cleanall: clean
	-$(RM) -r _x.* .Xil .run
	-$(RM) -r $(XCLBIN_DIR)
