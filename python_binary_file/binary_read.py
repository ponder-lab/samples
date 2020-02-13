import numpy as np

ar = np.empty([])
with open("odata.bin", "rb") as f:
    ndimb = f.read(4)
    ndims = np.frombuffer(ndimb, dtype=np.int32)

    shapeb = f.read(4*ndims[0])
    shape = np.frombuffer(shapeb, dtype=np.int32)

    shapet = ()
    for i in range(ndims[0]):
        shapet += (shape[i],)
    
    arb = f.read()
    ar = np.frombuffer(arb, dtype=np.float32)

    ar = ar.reshape(shapet)

print("Received data =")
print(ar)
