TARGET = my_add.so

$(TARGET): test.cpp
	g++ -o $@ -I/usr/include/python3.6m/ -fPIC -shared $^ -lboost_python3 -lboost_numpy3
#g++ -o $@ -I/usr/include/python3.6m/ -fPIC -shared $^ -lboost_python3

clean:
	rm $(TARGET)

