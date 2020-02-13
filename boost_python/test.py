import numpy as np
import my_add

a = my_add.my_add(1.0, 1.5)
print(a)

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
my_add.show_array(a)

a = [1.0, 2.0]
my_add.show_list(a)
