#!/bin/bash

# sudo apt install iverilog
# sudo apt install gtkwave

iverilog -o test_lut2 -s test_lut2 *.v
./test_lut2
gtkwave test_lut2.vcd
