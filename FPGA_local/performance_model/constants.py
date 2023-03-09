from common import *

D = 128
K = 256 # 256 row per LUT


""" A number of constant consumption components that should be added to PEs"""

# FIFO depth=512, width=32 bit
resource_FIFO_d512_w32 = Resource()
resource_FIFO_d512_w32.LUT = 54
resource_FIFO_d512_w32.FF = 95
resource_FIFO_d512_w32.BRAM_18K = 2 * 0.5
resource_FIFO_d512_w32.URAM = 0
resource_FIFO_d512_w32.DSP48E = 0
resource_FIFO_d512_w32.HBM_bank = 0

# depth 512, width = 512 = 16 * FIFO_d512_w32
resource_FIFO_d512_w512 = Resource()
resource_FIFO_d512_w512.LUT = 16 * 54
resource_FIFO_d512_w512.FF = 16 * 95
resource_FIFO_d512_w512.BRAM_18K = 16 * 2 * 0.5
resource_FIFO_d512_w512.URAM = 16 * 0
resource_FIFO_d512_w512.DSP48E = 16 * 0
resource_FIFO_d512_w512.HBM_bank = 16 * 0

# FIFO depth=2, width=8 bit
resource_FIFO_d2_w8 = Resource()
resource_FIFO_d2_w8.LUT = 20
resource_FIFO_d2_w8.FF = 6
resource_FIFO_d2_w8.BRAM_18K = 2 * 0
resource_FIFO_d2_w8.URAM = 0
resource_FIFO_d2_w8.DSP48E = 0
resource_FIFO_d2_w8.HBM_bank = 0

# FIFO depth=2, width=32 bit
resource_FIFO_d2_w32 = Resource()
resource_FIFO_d2_w32.LUT = 30
resource_FIFO_d2_w32.FF = 6
resource_FIFO_d2_w32.BRAM_18K = 2 * 0
resource_FIFO_d2_w32.URAM = 0
resource_FIFO_d2_w32.DSP48E = 0
resource_FIFO_d2_w32.HBM_bank = 0

# FIFO depth=2, width=512 bit
resource_FIFO_d2_w512 = Resource()
resource_FIFO_d2_w512.LUT = 484
resource_FIFO_d2_w512.FF = 964
resource_FIFO_d2_w512.BRAM_18K = 2 * 0
resource_FIFO_d2_w512.URAM = 0
resource_FIFO_d2_w512.DSP48E = 0
resource_FIFO_d2_w512.HBM_bank = 0
