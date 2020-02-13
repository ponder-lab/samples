TARGET = binary_proc

$(TARGET): binary_proc.cpp
	g++ -o $@ $^

clean:
	rm -rf $(TARGET) *.bin

run: $(TARGET)
	python binary_write.py
	./$(TARGET)
	python binary_read.py

