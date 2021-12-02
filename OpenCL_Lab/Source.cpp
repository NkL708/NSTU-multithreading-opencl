#include <iostream>
#include <string>
#include <fstream>
#include <CL/cl.hpp>

void fillCube(int*** cube, int k, int m, int n, int t1, int t2)
{
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < m; j++)
		{
			for (int k = 0; k < n; k++)
			{
				if (i == 0) 
				{
					cube[i][j][k] = t1;
					continue;
				}
				else if (i == (k - 1)) 
				{
					cube[i][j][k] = t2;
					continue;
				}
				cube[i][j][k] = rand() % 100;
			}
		}
	}
}

void printCube(int*** cube, int k, int m, int n, std::ofstream &file)
{
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < m; j++)
		{
			for (int k = 0; k < n; k++)
			{
				if (!((k + 1) % 15))
					file << std::endl;
				file << cube[i][j][k] << " ";
			}
		}
	}
	file << std::endl << std::endl;
}

// cmd param - k, m, n, t1, t2

int main(int argc, char* argv[])
{
	std::ofstream shuffledCube("shuffledCube.txt");
	int k = 5, m = 5, n = 5, t1 = 10, t2 = 15;
	int*** cube;
	if (argc > 1)
	{
		k = atoi(argv[1]), m = atoi(argv[2]), n = atoi(argv[3]),
			t1 = atoi(argv[4]), t2 = atoi(argv[5]);
	}
	// Linear
	cube = new int**[k];
	for (int i = 0; i < m; i++)
	{
		cube[i] = new int*[m];
		for (int j = 0; j < n; j++)
		{
			cube[i][j] = new int[n];
		}
	}
	fillCube(cube, k, m, n, t1, t2);
	printCube(cube, k, m, n, shuffledCube);

	// Init
	cl_platform_id platform;
	cl_device_id device;
	cl_int error = 0;
	std::ifstream file("program.cl");
	std::string fileText = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
	const char* srcText = fileText.data();
	size_t srcLength = fileText.size();
	// Get GPU
	error |= clGetPlatformIDs(1, &platform, NULL);
	error |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	// Compile and build
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
	cl_program program = clCreateProgramWithSource(context, 1, &srcText, &srcLength, &error);
	error |= clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	// What funtion from file we have to run
	cl_kernel kernel = clCreateKernel(program, "test", &error);
	// Add to Queue
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &error);
	// Result buffer
	cl_mem buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) /*size of result*/, NULL, NULL);
	// Invoke kernel
	//error |= clSetKernelArg(kernel, 0, sizeof(type), &param);
	//...
	size_t localSize[2] = { 1024, 1 };
	size_t globalSize[2] = { 1024, 1 };
	// Start task
	error |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
	// Read Result
	int*** result[1][1];
	//error |= clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(cl_uint) /*size of result*/, result, 0, NULL, NULL);
	// End all work and free memory
	clFinish(queue);
	clReleaseKernel(kernel);
	clReleaseMemObject(buffer);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}