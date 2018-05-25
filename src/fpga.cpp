#include "fpga.hpp"

// class BinaryReader
BinaryReader::BinaryReader(std::string path) {
    // Read FPGA binary
    int fd = open(path.c_str(), O_RDONLY);
    FILE *fp = fdopen(fd, "rb");
    struct stat stat_buf;

    fstat(fd, &stat_buf);

    size = (size_t)stat_buf.st_size;
    binary = (unsigned char *)malloc(size);	

    fread(binary, size, 1, fp);
    fclose(fp);
}

BinaryReader::~BinaryReader() {
    free(binary);
}


// class FPGAContext
FPGAContext::FPGAContext(std::string &context_name, std::vector<cl_device_id> &devs, cl_context_properties *properties,
              void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret) {
    devices = devs;
    context = clCreateContext(properties, devices.size(), devices.data(), pfn_notify, user_data, errcode_ret);
}

FPGAContext::~FPGAContext() {
    for (auto itr = programs.begin(); itr != programs.end(); itr++) {
        clReleaseProgram(itr->second);
    }
    if (context != NULL) {
        clReleaseContext(context);  
    }
}

cl_int FPGAContext::loadProgram(std::string &program_name, std::string &path, cl_int *binary_status) {
    BinaryReader binary = BinaryReader(path);

    // Create program
    cl_int errcode_ret;

    cl_program program = clCreateProgramWithBinary(context, devices.size(), devices.data(),
        &(binary.size), (const unsigned char **)&(binary.binary), binary_status, &errcode_ret);
 
    programs[program_name] = program;
    return errcode_ret;
}


// class FPGAHostMemory
FPGAHostMemory::FPGAHostMemory(size_t n) :
  size(n)
{
    posix_memalign(&mem, 64, n);
}

FPGAHostMemory::~FPGAHostMemory() {
    free(mem);
}

// class FPGABuffer
FPGABuffer::FPGABuffer(const FPGAContext &context, cl_mem_flags flags, size_t n, void *host_ptr, cl_int *errcode_ret) :
  objsize(sizeof(cl_mem)), length(n)
{
    mobj = clCreateBuffer(context.context, CL_MEM_READ_ONLY, length, host_ptr, errcode_ret);
}

FPGABuffer::~FPGABuffer() {
    clReleaseMemObject(mobj);
}


// class FPGAKernel
FPGAKernel::FPGAKernel(FPGAContext &context, std::string &program_name, std::string &kernel_name, cl_int *errcode_ret) {
    kernel = clCreateKernel(context.programs[program_name], kernel_name.c_str(), errcode_ret);
}

FPGAKernel::~FPGAKernel() {
    clReleaseKernel(kernel);
}

void FPGAKernel::setArg(cl_uint arg_index, FPGABuffer &buffer) {
    clSetKernelArg(kernel, arg_index, buffer.objsize, buffer.mobj);
}


// class FPGACommandQueue
FPGACommandQueue::FPGACommandQueue(const FPGAContext &context, cl_uint device_index, cl_command_queue_properties properties, cl_int *errcode_ret) {
    queue = clCreateCommandQueue(context.context, context.devices[device_index], properties, errcode_ret);
}

FPGACommandQueue::~FPGACommandQueue() {
    clReleaseCommandQueue(queue);
}


void FPGACommandQueue::waitCommits() {
    clFinish(queue);
}

void FPGACommandQueue::requestTask(FPGAKernel &kernel) {
    clEnqueueTask(queue, kernel.kernel, 0, NULL, NULL);
    clFinish(queue);
}

void FPGACommandQueue::commitTask(FPGAKernel &kernel) {
    clEnqueueTask(queue, kernel.kernel, 0, NULL, NULL);
}

void FPGACommandQueue::writeBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n) {
    clEnqueueWriteBuffer(queue, mobj.mobj, CL_TRUE, 0, n, mem.mem, 0, NULL, NULL);
    clFinish(queue);
}

void FPGACommandQueue::commitWriteBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n) {
    clEnqueueWriteBuffer(queue, mobj.mobj, CL_TRUE, 0, n, mem.mem, 0, NULL, NULL);
}

void FPGACommandQueue::readBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n) {
    clEnqueueReadBuffer(queue, mobj.mobj, CL_TRUE, 0, n, mem.mem, 0, NULL, NULL);
    clFinish(queue);
}

void FPGACommandQueue::commitReadBuffer(FPGAHostMemory &mem, FPGABuffer &mobj, size_t n) {
    clEnqueueReadBuffer(queue, mobj.mobj, CL_TRUE, 0, n, mem.mem, 0, NULL, NULL);
}


// class FPGA
FPGA::FPGA() {
    clGetPlatformIDs(0, NULL, &num_platforms);
    // Search for an openCL platform
    platforms.resize(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL); 

    for (auto itr = platforms.begin(); itr != platforms.end(); itr++) {
        cl_uint num_devices = 0;
        clGetDeviceIDs(*itr, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

        std::vector<cl_device_id> *v = new std::vector<cl_device_id>(num_devices);
        clGetDeviceIDs(*itr, CL_DEVICE_TYPE_ALL, num_devices, v->data(), NULL);

        devices[*itr] = v;
    }
}

FPGA::~FPGA() {
  for (auto itr = devices.begin(); itr != devices.end(); itr++) {
      delete itr->second;
  }
}

cl_int FPGA::loadBinaryProgram(std::string &context_name, std::string &program_name, std::string &path, cl_int *binary_status) {
    auto itr = contexts.find(context_name);
    if (itr == contexts.end()) {
        printf("no such context.\n");
        return 0;
    }

    FPGAContext *target = contexts[context_name];
    return target->loadProgram(program_name, path, binary_status);
}

FPGAContext &FPGA::createContext(std::string &context_name, std::vector<cl_device_id> &devs, cl_context_properties *properties,
              void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret) {
    auto itr = contexts.find(context_name);
    if (itr != contexts.end()) {
        throw std::exception();
    }

    FPGAContext *new_context = new FPGAContext(context_name, devs, properties, pfn_notify, user_data, errcode_ret);
    contexts[context_name] = new_context;
    return *(new_context);
}

FPGAContext &FPGA::getContext(std::string context_name) {
    return *(contexts[context_name]);
}

cl_platform_id FPGA::getFirstPlatformID() {
    return *(platforms.begin());  
}

cl_device_id FPGA::getFirstDeviceID() {
    return *(devices[getFirstPlatformID()]->begin());
}

