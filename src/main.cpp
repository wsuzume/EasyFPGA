#include <stdio.h>	
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
#include <time.h>

#include "fpga.hpp"

// High-resolution timer.
double getCurrentTimestamp() 
{
    timespec a;
    clock_gettime(CLOCK_MONOTONIC, &a);
    return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
}


const size_t N = 1024 * 1024 * 64;
const size_t MEM_SIZE = sizeof(float) * N;

int main()
{ 

    // get platformIDs and deviceIDs
    FPGA fpga = FPGA();

    // context name(any name you like, to specify the context)
    std::string context_name("Context1");
    // push deviceIDs to vector that you want to add the context
    std::vector<cl_device_id> devices;
    devices.push_back(fpga.getFirstDeviceID());
    // create context
    FPGAContext &context = fpga.createContext(context_name, devices, NULL, NULL, NULL, NULL);

    // program name(any name you like, to specify the program)
    std::string program_name("Program1");
    std::string program_path("./bin/program_name.aocx");
    context.loadProgram(program_name, program_path, NULL);

    // create host side memory
    FPGAHostMemory din1 = FPGAHostMemory(MEM_SIZE);
    FPGAHostMemory din2 = FPGAHostMemory(MEM_SIZE);
    FPGAHostMemory dout = FPGAHostMemory(MEM_SIZE);
    
    // create device side memory
    FPGABuffer mdin1 = FPGABuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, NULL);
    FPGABuffer mdin2 = FPGABuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, NULL);
    FPGABuffer mdout = FPGABuffer(context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, NULL);

    // kernel name(the same name to the kernel code)
    std::string kernel_name("kernel_name");
    FPGAKernel kernel = FPGAKernel(context, program_name, kernel_name, NULL);
    kernel.setArg(0, mdin1);
    kernel.setArg(1, mdin2);
    kernel.setArg(2, mdout);

    // create command queue
    FPGACommandQueue queue = FPGACommandQueue(context, 0, 0, NULL);

    // Write data to FPGA
    queue.writeBuffer(din1, mdin1, MEM_SIZE);
    queue.writeBuffer(din2, mdin2, MEM_SIZE);
    // execute kernel
    queue.requestTask(kernel);
    // read data from FPGA
    queue.readBuffer(dout, mdout, MEM_SIZE);

    return 0;
}


