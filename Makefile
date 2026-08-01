# =============================================================================
# fpga-nn-accel -- simulation
#
#   make test        run every testbench, report pass/fail
#   make tb_core     run one testbench
#   make clean
#
# Run from the repo root: the testbenches load mem/*.mem through $readmemh,
# which resolves relative to the working directory. To build elsewhere, pass
# an absolute path:
#
#   make test MEM_DIR=/abs/path/to/mem
#
# IVERILOG/VVP are overridable for a local install:
#   make test IVERILOG=~/iv/usr/bin/iverilog VVP=~/iv/usr/bin/vvp
# =============================================================================

IVERILOG ?= iverilog
VVP      ?= vvp
BUILD    ?= build
MEM_DIR  ?=

IVFLAGS := -g2005
ifneq ($(MEM_DIR),)
IVFLAGS += -DMEM_DIR='"$(MEM_DIR)"'
endif

RTL       := rtl/nn_accel_core.v rtl/rom_sync.v rtl/mac_array.v rtl/mac_unit.v \
             rtl/requantize.v rtl/argmax.v
TESTS     := tb_core tb_img_loader tb_mac_array tb_mac_unit tb_datapath \
             tb_uart_rx tb_uart_tx tb_top_uart

# sources per testbench
SRC_tb_core        := tb/tb_core.v $(RTL)
SRC_tb_img_loader  := tb/tb_img_loader.v rtl/img_loader.v
SRC_tb_mac_array   := tb/tb_mac_array.v rtl/mac_array.v rtl/mac_unit.v
SRC_tb_mac_unit    := tb/tb_mac_unit.v rtl/mac_unit.v
SRC_tb_datapath    := tb/tb_datapath.v rtl/requantize.v rtl/argmax.v
SRC_tb_uart_rx     := tb/tb_uart_rx.v rtl/uart_rx.v
SRC_tb_uart_tx     := tb/tb_uart_tx.v rtl/uart_tx.v
SRC_tb_top_uart    := tb/tb_top_uart.v rtl/nn_accel_top.v rtl/nn_accel_core.v \
                      rtl/img_loader.v rtl/uart_rx.v rtl/uart_tx.v \
                      rtl/seven_seg.v rtl/rom_sync.v rtl/mac_array.v \
                      rtl/mac_unit.v rtl/requantize.v rtl/argmax.v

.PHONY: test $(TESTS) clean lint mem

# A testbench that fails prints FAIL lines but still exits 0, so the pass/fail
# decision comes from the summary line each one prints, not the exit status.
test:
	@mkdir -p $(BUILD)
	@fail=0; \
	for t in $(TESTS); do \
	  $(MAKE) -s $$t > $(BUILD)/$$t.log 2>&1; \
	  if grep -q "ALL TESTS PASSED" $(BUILD)/$$t.log; then \
	    n=$$(grep -c '^pass' $(BUILD)/$$t.log); \
	    printf "  \033[32mPASS\033[0m  %-16s %3d checks\n" $$t $$n; \
	  else \
	    printf "  \033[31mFAIL\033[0m  %-16s see $(BUILD)/$$t.log\n" $$t; \
	    grep -m3 -E "FAIL|error" $(BUILD)/$$t.log | sed 's/^/          /'; \
	    fail=1; \
	  fi; \
	done; \
	echo; \
	if [ $$fail -eq 0 ]; then echo "  all testbenches passed"; \
	else echo "  FAILURES -- see $(BUILD)/*.log"; exit 1; fi

# Build and run one testbench: make tb_core
$(TESTS):
	@mkdir -p $(BUILD)
	@$(IVERILOG) $(IVFLAGS) -o $(BUILD)/$@.vvp $(SRC_$@)
	@$(VVP) $(BUILD)/$@.vvp

# Elaborate the Basys 3 top level without running it -- catches port and
# width errors that no testbench covers.
lint:
	@mkdir -p $(BUILD)
	@$(IVERILOG) $(IVFLAGS) -o $(BUILD)/top.vvp -s nn_accel_top \
	  rtl/nn_accel_top.v rtl/nn_accel_core.v rtl/img_loader.v rtl/uart_rx.v \
	  rtl/uart_tx.v rtl/seven_seg.v rtl/rom_sync.v rtl/mac_array.v \
	  rtl/mac_unit.v rtl/requantize.v rtl/argmax.v
	@echo "  nn_accel_top elaborates clean"

# Regenerate the weight/golden files from the trained checkpoint.
#
# NEEDS data/sign_mnist_*.csv, which is NOT in the repo (164 MB) -- that is why
# mem/*.mem is tracked rather than gitignored: without the dataset a fresh
# clone cannot rebuild it, and tb_img_loader.v and the web server both read
# those files. Training itself is deliberately not wired in here; it is slow,
# needs a GPU to be pleasant, and overwrites the checkpoint.
PYTHON ?= ./venv/bin/python

mem:
	@test -f data/sign_mnist_train.csv || { \
	  echo "  data/sign_mnist_train.csv missing -- download Sign Language MNIST"; \
	  echo "  (kaggle.com/datasets/datamunge/sign-language-mnist) into data/"; \
	  exit 1; }
	$(PYTHON) python/quantize.py
	$(PYTHON) python/pack_mem.py
	$(PYTHON) python/golden_check.py
	@echo "  mem/ regenerated -- run 'make test' to confirm the RTL still matches"

clean:
	@rm -rf $(BUILD)
