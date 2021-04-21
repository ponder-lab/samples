module test_lut2;

   reg [1:0] in;
   reg [3:0] param;
   wire      oval;

   integer   i, j;
   
   lut2 lut2_(in, param, oval);

   initial
     begin
        $dumpfile("test_lut2.vcd");
        $dumpvars(0, test_lut2);
        
        in = 0;
        param = 0;
             
        for(j = 0; j < 16; j += 1)
          begin
             for(i = 0; i < 4; i += 1)
               begin
                  #1;
                  in = i;
                  param = j;

                  #9;
                  $display("in=%b param=%b out=%b", in, param, oval);
               end
          end

        #10;
        $finish;
     end
endmodule // test_lut2
