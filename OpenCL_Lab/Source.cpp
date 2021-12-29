#include <iostream>
#include <string>
#include <fstream>

#include <math.h>
#include <omp.h>
#include <CL/cl.hpp>

void copy(float* target, float* src, int size)
{
	for (int i = 0; i < size; i++) {
		target[i] = src[i];
	}
}

void fillCuboid(float* cuboid, int k, int m, int n, int temp1, int temp2)
{
	for (int z = 0; z < k; z++) {
		for (int y = 0; y < m; y++) {
			for (int x = 0; x < n; x++) {
				int index = (z * m * n) + (y * n) + x;
				if (z == 0) {
					cuboid[index] = 0;
				}	
				else if (z == k - 1 && x == 0) {
					cuboid[index] = (float)temp1;
				}
				else if (z == k - 1 && x == n - 1) {
					cuboid[index] = (float)temp2;
				}
				else {  // from -100 to +100
					cuboid[index] = (rand() % 201) - 100.f;
				}		
			}
		}
	}
}

void printCuboidToFile(float* cuboid, int k, int m, int n, std::ofstream &file)
{
	for (int z = 0; z < k; z++) {
		for (int y = 0; y < m; y++) {
			for (int x = 0; x < n; x++) {
				int index = (z * m * n) + (y * n) + x;
				file << cuboid[index] << " ";
			}
			file << "\n";
		}
		file << "\n\n";
	}
}

float countAverage(float* cuboid, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += cuboid[i];
	}
	return sum / size;
}

float* distribute(float* cuboid, int k, int m, int n)
{
	int size = k * m * n;
	float* result = new float[size];
	copy(result, cuboid, size);
	bool isDissipated = false;
	// Ends if temperatures in cube becomes balanced
	while (!isDissipated) {
		int dissipatedCount = 0;
		for (int z = 0; z < k; z++) {
			for (int y = 0; y < m; y++) {
				for (int x = 0; x < n; x++) {
					// Calc average temperature
					int index = (z * m * n) + (y * n) + x;
					float sum = 0;
					int count = 0;
					float average;
					for (int zSum = z - 1; zSum <= z + 1; zSum++) {
						for (int ySum = y - 1; ySum <= y + 1; ySum++) {
							for (int xSum = x - 1; xSum <= x + 1; xSum++) {
								if (zSum >= 0 && ySum >= 0 && xSum >= 0
									&& zSum < k && ySum < m && xSum < n) {
									count++;
									sum += result[(zSum * m * n) + (ySum * n) + xSum];
								}
							}
						}
					}
					average = round(sum / count * 100) / 100;
					if (average == result[index]) {
						dissipatedCount++;
					}
					else {
						result[index] = average;
					}
				}
			}
		}
		if (dissipatedCount == size) {
			isDissipated = true;
		}
	}
	return result;
}

float* distributeOpenCL(float* cuboid, int k, int m, int n)
{
	// OpenCL init
	int size = k * m * n;
	float* hResult = new float[size];
	cl_platform_id platform;
	cl_device_id device;
	cl_int error = 0;
	std::ifstream file("program.cl");
	std::string fileText = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
	const char* srcText = fileText.data();
	size_t srcLength = fileText.size();
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue queue;
	cl_mem dCuboid, dRes;
	size_t localSize[1] = { k * m };
	// ceil(size / (float)localSize[0]) * localSize[0], ceil(size / (float)localSize[1]) * localSize[1]
	size_t globalSize[1] = { k * m };
	// Get GPU
	error |= clGetPlatformIDs(1, &platform, NULL);
	error |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	// Compile and build
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
	program = clCreateProgramWithSource(context, 1, &srcText, &srcLength, &error);
	error |= clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	// What funtion from file we have to run
	kernel = clCreateKernel(program, "distributeKernel", &error);
	// Add to Queue
	queue = clCreateCommandQueue(context, device, NULL, &error);
	// Create buffer
	dCuboid = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, NULL);
	dRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, NULL);
	// Write data to buffer
	error |= clEnqueueWriteBuffer(queue, dCuboid, CL_TRUE, 0, sizeof(float) * size, cuboid, 0, NULL, NULL);
	// Kernel args
	error |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &dCuboid);
	error |= clSetKernelArg(kernel, 1, sizeof(int), &k);
	error |= clSetKernelArg(kernel, 2, sizeof(int), &m);
	error |= clSetKernelArg(kernel, 3, sizeof(int), &n);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &dRes);
	// Start task
	error |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
	// Wait execution
	clFinish(queue);
	// Read Result
	error |= clEnqueueReadBuffer(queue, dRes, CL_TRUE, 0, sizeof(float) * size, hResult, 0, NULL, NULL);
	// Deallocation
	clReleaseKernel(kernel);
	clReleaseMemObject(dCuboid);
	clReleaseMemObject(dRes);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return hResult;
}

int main(int argc, char* argv[])
{
	std::ofstream filledFile("filled.txt");
	std::ofstream resLFile("resL.txt");
	std::ofstream resPFile("resP.txt");
	double durationL, durationP, time1, time2;
	int k, m, n, temp1, temp2, size;
	float* cuboid, * resL, * resP;
	// max - 32
	if (argc > 1) {
		k = atoi(argv[1]), m = atoi(argv[2]), n = atoi(argv[3]),
			temp1 = atoi(argv[4]), temp2 = atoi(argv[5]);
	}
	else {
		k = 5, m = 5, n = 5, temp1 = 10, temp2 = 15;
	}
	size = k * m * n;
	// Linear
	cuboid = new float[size];
	fillCuboid(cuboid, k, m, n, temp1, temp2);
	std::cout << "Average - " << countAverage(cuboid, size) << std::endl;
	printCuboidToFile(cuboid, k, m, n, filledFile);
	time1 = omp_get_wtime();
	resL = distribute(cuboid, k, m, n);
	time2 = omp_get_wtime();
	durationL = time2 - time1;
	printCuboidToFile(resL, k, m, n, resLFile);
	// Parallel
	time1 = omp_get_wtime();
	resP = distributeOpenCL(cuboid, k, m, n);
	time2 = omp_get_wtime();
	durationP = time2 - time1;
	printCuboidToFile(resP, k, m, n, resPFile);
	std::cout << "Linear time: " << durationL << std::endl;
	std::cout << "Parallel time: " << durationP << std::endl;
	std::cout << "Parallel faster than linear on: " << durationL - durationP << std::endl;
	// Delete 3d arrays, closing files
	delete[] cuboid; delete[] resL; delete[] resP;
	filledFile.close(); resLFile.close(); resPFile.close();
	return 0;
}