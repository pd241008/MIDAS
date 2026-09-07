# STM32F407VG TFLM firmware Makefile
# Builds against the upstream microlite library (libtensorflow-microlite.a)
#
# Usage:
#   make          — build firmware (produces build/mcu_fw.elf + .bin + .hex + .map)
#   make flash    — flash via st-flash (requires ST-LINK USB attached to WSL)
#   make size     — show flash/RAM usage (arm-none-eabi-size)
#   make clean    — remove build artifacts

# ---- Toolchain ----
PREFIX  ?= arm-none-eabi-
CC       = $(PREFIX)gcc
CXX      = $(PREFIX)g++
AS       = $(PREFIX)gcc -x assembler-with-cpp
LD       = $(PREFIX)g++
OBJCOPY  = $(PREFIX)objcopy
SIZE     = $(PREFIX)size

# ---- Paths ----
# TFLM_ROOT points to the tflite-micro repo (full checkout with built library)
TFLM_ROOT ?= /tmp/opencode/tflm/tflite-micro

# CMSIS Core headers (for core_cm4.h, cmsis_compiler.h, cmsis_gcc.h)
CMSIS_CORE  = $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/Core/Include

# Prebuilt microlite library for cortex-m4+fp
MICROLITE_A = $(TFLM_ROOT)/gen/cortex_m_generic_cortex-m4+fp_default_gcc/lib/libtensorflow-microlite.a

# TFLM include roots
TFLM_INC  = $(TFLM_ROOT)
TFLM_INC += $(TFLM_ROOT)/tensorflow
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/micro
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/kernels/internal
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/core/c
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/schema
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/gemmlowp
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/ruy
TFLM_INC += $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/pigweed

# ---- Build dir ----
BUILDDIR = build

# ---- Flags ----
ARCH_FLAGS = -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb

CFLAGS  = $(ARCH_FLAGS) -Wall -Wextra -O3 -g0
CFLAGS += -ffunction-sections -fdata-sections -fno-common
CFLAGS += -DTF_LITE_STATIC_MEMORY
CFLAGS += -DTF_LITE_DISABLE_X86_NEON
CFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS

CXXFLAGS = $(CFLAGS) -fno-rtti -fno-exceptions -std=c++17

ASFLAGS  = $(ARCH_FLAGS) -Wall

LDFLAGS  = $(ARCH_FLAGS) -Wl,--gc-sections -Wl,-Map=$(BUILDDIR)/mcu_fw.map
LDFLAGS += -T f407vg.ld -nostartfiles -specs=nosys.specs -specs=nano.specs
LDFLAGS += -lm

# ---- Source files ----
SRCStartup = src/startup_stm32f40xx.s
SRCSysInit = src/system_init.c
SRCMain    = src/main.cc

OBJStartup = $(BUILDDIR)/startup.o
OBJSysInit = $(BUILDDIR)/system_init.o
OBJMain    = $(BUILDDIR)/main.o
OBJS       = $(OBJStartup) $(OBJSysInit) $(OBJMain)

# ---- Target ----
TARGET = $(BUILDDIR)/mcu_fw

.PHONY: all flash size clean

all: $(TARGET).elf $(TARGET).bin $(TARGET).hex size

# ---- Include paths ----
INCLUDES  = -Isrc
INCLUDES += -I$(CMSIS_CORE)
$(foreach d,$(TFLM_INC),$(eval INCLUDES += -I$(d)))

# ---- Compilation rules (explicit, avoids GNU Make double-dot pattern bug) ----
$(OBJStartup): $(SRCStartup)
	@mkdir -p $(dir $@)
	$(AS) $(ASFLAGS) $(INCLUDES) -c $< -o $@

$(OBJSysInit): $(SRCSysInit)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJMain): $(SRCMain)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ---- Link ----
$(TARGET).elf: $(OBJS)
	$(LD) $(OBJS) $(MICROLITE_A) $(LDFLAGS) -o $@

# ---- Binary outputs ----
$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex $< $@

# ---- Flash via st-flash ----
flash: $(TARGET).bin
	st-flash write $< 0x08000000

# ---- Size report ----
size: $(TARGET).elf
	@echo ""
	@echo "=== Firmware size breakdown ==="
	$(SIZE) $<
	@echo ""
	@echo "=== Component sizes (approximate) ==="
	@echo "  Startup (.isr_vector + .text):"
	$(SIZE) --format=sysv $(OBJStartup) 2>/dev/null || true
	@echo "  system_init.c:"
	$(SIZE) --format=sysv $(OBJSysInit) 2>/dev/null || true
	@echo "  main.cc:"
	$(SIZE) --format=sysv $(OBJMain) 2>/dev/null || true
	@echo "  microlite library:"
	$(SIZE) --format=sysv $(MICROLITE_A) 2>/dev/null | grep -E "TOTAL|microlite" | head -3 || true

# ---- Clean ----
clean:
	rm -rf $(BUILDDIR)
