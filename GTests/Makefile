CC      = nvcc
TARGET  = test
C_FILES = 
#$(filter-out $(TARGET).cpp, $(wildcard *.cpp))
OBJS    = 
#$(patsubst %.cpp,%.o,$(C_FILES))
CFLAGS  = 
#-g -Wall -Werror -pedantic-errors -std=c++14
LDFLAGS = -lgtest 
#-pthread 
#-lgtest_main 

.PHONY: all clean
all: $(TARGET)
$(TARGET): $(OBJS) $(TARGET).cu
	$(CC) $(CFLAGS) $(OBJS) $(TARGET).cu -o $(TARGET) $(LDFLAGS)
%.o: %.cpp %.h
	$(CC) $(CFLAGS) -c -o $@ $<
clean:
	rm -f $(OBJS) $(TARGET) $(TARGET).exe

