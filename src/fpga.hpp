#pragma once

#include <cstdio>
#include <cstdarg>
#include <vector>
#include <map>
#include <string>
#include <iostream>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <CL/opencl.h>

class BinaryReader {
  public:
    size_t size;
    unsigned char *binary;
    BinaryReader(std::string path);
    ~BinaryReader();
};


class FPGAContext {
  public:
    cl_context context;

    std::vector<cl_device_id> devices;
    std::map<std::string, cl_program> programs;

    std::vector<cl_mem> buffers;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues; 

    FPGAContext(std::string &context_name, std::vector<cl_device_id> &devs, cl_context_properties *properties,
                  void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret);
    ~FPGAContext();
    cl_int loadProgram(std::string &program_name, std::string &path, cl_int *binary_status);

};


class FPGAHostMemory {
  public:
    size_t size;
	  void *mem;

    FPGAHostMemory(size_t n);
    ~FPGAHostMemory();
};

class FPGABuffer {
  public:
    size_t objsize;
    size_t length;
    cl_mem mobj;

    FPGABuffer(const FPGAContext &context, cl_mem_flags flags, size_t n, void *host_ptr, cl_int *errcode_ret);
    ~FPGABuffer();
};

class FPGAKernel {
  public:
    cl_kernel kernel;

    FPGAKernel(FPGAContext &context, std::string &program_name, std::string &kernel_name, cl_int *errcode_ret);
    ~FPGAKernel();
    void setArg(cl_uint arg_index, FPGABuffer &buffer);
};

class FPGACommandQueue {
  public:
    cl_command_queue queue;

    FPGACommandQueue(const FPGAContext &context, cl_uint device_index, cl_command_queue_properties properties, cl_int *errcode_ret);
    ~FPGACommandQueue();

    void waitCommits();
    void requestTask(FPGAKernel &kernel);
    void commitTask(FPGAKernel &kernel);
    void writeBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n);
    void commitWriteBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n);
    void readBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n);
    void commitReadBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n);
};

class FPGA {
  public:
    cl_uint num_platforms;
    std::vector<cl_platform_id> platforms;
    std::map<cl_platform_id, std::vector<cl_device_id>*> devices;
    std::map<std::string, FPGAContext*> contexts;

    FPGA();
    ~FPGA();
    cl_int loadBinaryProgram(std::string &context_name, std::string &program_name, std::string &path, cl_int *binary_status);
    FPGAContext &createContext(std::string &context_name, std::vector<cl_device_id> &devs, cl_context_properties *properties,
                  void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret);
    FPGAContext &getContext(std::string context_name);
    cl_platform_id getFirstPlatformID();
    cl_device_id getFirstDeviceID();
};
