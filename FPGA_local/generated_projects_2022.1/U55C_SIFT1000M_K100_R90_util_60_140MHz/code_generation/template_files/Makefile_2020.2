############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO)  "  make test DEVICE=<FPGA platform>"
	$(ECHO)  "     Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	$(ECHO)  ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""

############################## Setting up Project Variables ##############################

TARGET := hw
HOST_ARCH := x86
SYSROOT := 

# include ./utils.mk
### utils.mk ###
#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------

PROFILE := yes

#Generates profile summary report
ifeq ($(PROFILE), yes)
VPP_LDFLAGS += --profile.data all:all:all
endif

DEBUG := no
B_TEMP = `$(XF_PROJ_ROOT)/common/utility/parse_platform_list.py $(DEVICE)`

#Generates debug summary report
ifeq ($(DEBUG), yes)
VPP_LDFLAGS += --dk list_ports
endif

#Setting Platform Path
ifeq ($(findstring xpfm, $(DEVICE)), xpfm)
	B_NAME = $(shell dirname $(DEVICE))
else
	B_NAME = $(B_TEMP)/$(DEVICE)
endif

#Checks for XILINX_VITIS
check-vitis:
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif

#Checks for XILINX_XRT
check-xrt:
ifeq ($(HOST_ARCH), x86)
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly and rerun)
endif
else
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif
endif

#Checks for Correct architecture
ifneq ($(HOST_ARCH), $(filter $(HOST_ARCH),aarch64 aarch32 x86))
$(error HOST_ARCH variable not set, please set correctly and rerun)
endif

#Setting CXX
CXX := g++

#Checks for EDGE_COMMON_SW
ifneq ($(HOST_ARCH), x86)
ifndef EDGE_COMMON_SW
$(error EDGE_COMMON_SW variable is not set, please set correctly and rerun)
endif
ifeq ($(HOST_ARCH), aarch64)
SYSROOT := $(EDGE_COMMON_SW)/sysroots/cortexa72-cortexa53-xilinx-linux
SD_IMAGE_FILE := $(EDGE_COMMON_SW)/Image
CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
else ifeq ($(HOST_ARCH), aarch32)
SYSROOT := $(EDGE_COMMON_SW)/sysroots/cortexa9t2hf-neon-xilinx-linux-gnueabi/
SD_IMAGE_FILE := $(EDGE_COMMON_SW)/uImage
CXX := $(XILINX_VITIS)/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin/arm-linux-gnueabihf-g++
endif
endif

gen_run_app:
ifneq ($(HOST_ARCH), x86)
	rm -rf run_app.sh
	$(ECHO) 'export LD_LIBRARY_PATH=/mnt:/tmp:$$LD_LIBRARY_PATH' >> run_app.sh
	$(ECHO) 'export PATH=$$PATH:/sbin' >> run_app.sh
	$(ECHO) 'export XILINX_XRT=/usr' >> run_app.sh
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(ECHO) 'export XILINX_VITIS=$$PWD' >> run_app.sh
	$(ECHO) 'export XCL_EMULATION_MODE=$(TARGET)' >> run_app.sh
endif
	$(ECHO) '$(EXECUTABLE) vadd.xclbin' >> run_app.sh
	$(ECHO) 'return_code=$$?' >> run_app.sh
	$(ECHO) 'if [ $$return_code -ne 0 ]; then' >> run_app.sh
	$(ECHO) 'echo "ERROR: host run failed, RC=$$return_code"' >> run_app.sh
	$(ECHO) 'fi' >> run_app.sh
	$(ECHO) 'echo "INFO: host run completed."' >> run_app.sh
endif
check-devices:
ifndef DEVICE
	$(error DEVICE not set. Please set the DEVICE properly and rerun. Run "make help" for more details.)
endif

#   device2xsa - create a filesystem friendly name from device name
#   $(1) - full name of device
device2xsa = $(strip $(patsubst %.xpfm, % , $(shell basename $(DEVICE))))

############################## Deprecated Checks and Running Rules ##############################
check:
	$(ECHO) "WARNING: \"make check\" is a deprecated command. Please use \"make run\" instead"
	make run

exe:
	$(ECHO) "WARNING: \"make exe\" is a deprecated command. Please use \"make host\" instead"
	make host

# Cleaning stuff
RM = rm -f
RMDIR = rm -rf

ECHO:= @echo

docs: README.rst

README.rst: description.json
	$(XF_PROJ_ROOT)/common/utility/readme_gen/readme_gen.py description.json
### utils.mk ###

XSA := 
ifneq ($(DEVICE), )
XSA := $(call device2xsa, $(DEVICE))
endif
TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

# SoC variables
RUN_APP_SCRIPT = ./run_app.sh
PACKAGE_OUT = ./package.$(TARGET)

LAUNCH_EMULATOR = $(PACKAGE_OUT)/launch_$(TARGET).sh
RESULT_STRING = TEST PASSED

VPP := v++
VPP_PFLAGS := 
CMD_ARGS = $(BUILD_DIR)/vadd.xclbin
SDCARD := sd_card

# Definition of include file locations
ifneq ($(HOST_ARCH), x86)
    xrt_path = $(SYSROOT)/usr
    OPENCL_INCLUDE:= $(xrt_path)/include/xrt
else
    xrt_path = $(XILINX_XRT)
    OPENCL_INCLUDE:= $(xrt_path)/include
endif

VIVADO_INCLUDE:= $(XILINX_VIVADO)/include
opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)
OPENCL_LIB:= $(xrt_path)/lib
opencl_LDFLAGS=-L$(OPENCL_LIB) -lOpenCL -pthread

CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++1y
LDFLAGS += $(opencl_LDFLAGS)

ifeq ($(findstring nodma, $(DEVICE)), nodma)
$(error [ERROR]: This example is not supported for $(DEVICE).)
endif

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I./includes/xcl2
HOST_SRCS += ./includes/xcl2/xcl2.cpp ./src/host.cpp 
# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps 
ifneq ($(TARGET), hw)
	VPP_FLAGS += -g
endif



# Adding config files to linker
VPP_LDFLAGS_vadd += --config design.cfg 
EXECUTABLE = ./host
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

############################## Declaring Binary Containers ##############################
BINARY_CONTAINERS += $(BUILD_DIR)/vadd.xclbin
BINARY_CONTAINER_vadd_OBJS += $(TEMP_DIR)/vadd.xo

############################## Setting Targets ##############################
CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig sd_card

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis $(BINARY_CONTAINERS)

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
$(TEMP_DIR)/vadd.xo: src/*.cpp src/*.hpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k vadd --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(BUILD_DIR)/vadd.xclbin: $(BINARY_CONTAINER_vadd_OBJS)
	mkdir -p $(BUILD_DIR)
ifeq ($(HOST_ARCH), x86)
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR)  $(VPP_LDFLAGS_vadd) -o'$(BUILD_DIR)/vadd.link.xclbin' $(+)
	$(VPP) -p $(BUILD_DIR)/vadd.link.xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/vadd.xclbin
else
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR) -o'$(BUILD_DIR)/vadd.xclbin' $(+)
endif

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | check-xrt
		$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(LAUNCH_EMULATOR) -run-app $(RUN_APP_SCRIPT) | tee run_app.log; exit $${PIPESTATUS[0]}
endif
else
ifeq ($(HOST_ARCH), x86)
	$(EXECUTABLE) $(CMD_ARGS)
endif
endif


.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(LAUNCH_EMULATOR) -run-app $(RUN_APP_SCRIPT) | tee run_app.log; exit $${PIPESTATUS[0]}
endif
else
ifeq ($(HOST_ARCH), x86)
	$(EXECUTABLE) $(CMD_ARGS)
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif


############################## Preparing sdcard ##############################
sd_card: $(BINARY_CONTAINERS) $(EXECUTABLE) gen_run_app
ifneq ($(HOST_ARCH), x86)
	$(VPP) $(VPP_PFLAGS) -p $(BUILD_DIR)/vadd.xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(SD_IMAGE_FILE) --package.sd_file xrt.ini --package.sd_file $(RUN_APP_SCRIPT) --package.sd_file $(EXECUTABLE) -o vadd.xclbin
endif

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb .ipcache

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin
