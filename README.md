# GLASS

## Usage
```#include "./GLASS/GLASS.cuh"```

## Setup
The Class DeviceTest creates 6 arrays, three on the host and three on the device. Any tests part of the class DeviceTest can use these arrays in their tests
The printArray method prints an integer array
The global_glass.cuh file call all of the library functions while using global header so they can be called on the host. All the method names are the same except are preceded with “global_”
## Making New Tests
Make a new method with the signature TEST_F(DeviceTest, <test name>). You can use the 6 device and host variables this way
Use EXPECT_EQ or ASSERT_EQ statements for the results of matrix operations. For more information on GoogleTest tests, see here http://google.github.io/googletest/primer.html
Ensure the following lines are in the main method 
::testing::InitGoogleTest();
 return RUN_ALL_TESTS();
When compiling use the library flag -lgtest

