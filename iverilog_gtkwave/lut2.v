module lut2(in, param, out);
   input [1:0] in;
   input [3:0] param;
   output      out;
   reg         out;
   
   always @*
     begin
        case(in)
          2'b00: out = param[0];
          2'b01: out = param[1];
          2'b10: out = param[2];
          2'b11: out = param[3];
          default: out = 1'bx;
        endcase // case (in)
     end

endmodule
   
