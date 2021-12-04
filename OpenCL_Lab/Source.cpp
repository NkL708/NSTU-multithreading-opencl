#include <iostream>
#include <string>
#include <fstream>

#include <omp.h>
#include <CL/cl.hpp>

float*** initCuboid(int k, int m, int n) 
{
	float*** result = new float** [k];
	for (int z = 0; z < k; z++) {
		result[z] = new float* [m];
		for (int y = 0; y < m; y++) {
			result[z][y] = new float[n];
		}
	}
	return result;
}

void deleteCuboid(float*** cuboid, int k, int m, int n) 
{
	for (int z = 0; z < k; z++) {
		for (int y = 0; y < m; y++) {
			delete[] cuboid[z][y];
		}
		delete[] cuboid[z];
	}
	delete[] cuboid;
}

void copy(float*** target, float*** src, int k, int m, int n)
{
	for (int z = 0; z < k; z++) {
		for (int y = 0; y < m; y++) {
			for (int x = 0; x < n; x++) {
				target[z][y][x] = src[z][y][x];
			}
		}
	}
}

void fillCuboid(float*** cuboid, int k, int m, int n, int temp1, int temp2)
{
	for (int z = 0; z < k; z++) {
		for (int y = 0; y < m; y++) {
			for (int x = 0; x < n; x++) {

				if (z == 0) {
					cuboid[z][y][x] = 0;
				}	
				else if (z == k - 1 && x == 0) {
					cuboid[z][y][x] = (float)temp1;
				}
				else if (z == k - 1 && x == n - 1) {
					cuboid[z][y][x] = (float)temp2;
				}
				else {  // -100 to +100
					cuboid[z][y][x] = (rand() % 201) - 100.f;
				}		
			}
		}
	}
}

void printCuboidToFile(float*** cuboid, int k, int m, int n, std::ofstream &file)
{
	for (int z = 0; z < k; z++) {
		for (int y = 0; y < m; y++) {
			for (int x = 0; x < n; x++) {
				file << cuboid[z][y][x] << " ";
			}
			file << "\n";
		}
		file << "\n\n";
	}
}

float*** distribute(float*** cuboid, int k, int m, int n)
{
	float*** result = initCuboid(k, m, n);
	copy(result, cuboid, k, m, n);
	bool isDissipated = false;
	int size = k * m * n;
	// Ends if temperatures in cube becomes balanced
	while (!isDissipated) {
		int dissipatedCount = 0;
		for (int z = 0; z < k; z++) {
			for (int y = 0; y < m; y++) {
				for (int x = 0; x < n; x++) {
					// Calc average temperature
					float sum = 0;
					int count = 0;
					float average;
					for (int zSum = z - 1; zSum <= z + 1; zSum++) {
						for (int ySum = y - 1; ySum <= y + 1; ySum++) {
							for (int xSum = x - 1; xSum <= x + 1; xSum++) {
								if (zSum >= 0 && ySum >= 0 && xSum >= 0
									&& zSum < k && ySum < m && xSum < n) {
									count++;
									sum += result[zSum][ySum][xSum];
								}
							}
						}
					}
					average = round(sum / count * 100) / 100;
					if (average == result[z][y][x]) {
						dissipatedCount++;
					}
					else {
						result[z][y][x] = average;
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

float*** distributeOpenCL(float*** cuboid, int k, int m, int n)
{
	// OpenCL init
	int size = k * m * n;
	float*** hResult = initCuboid(k, m, n);
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
	size_t localSize[2] = { 256, 1 };
	// ceil(size / (float)localSize[0]) * localSize[0], ceil(size / (float)localSize[1]) * localSize[1]
	size_t globalSize[2] = { 1024, 1 };
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
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &error);
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
	error |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
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
	int k = 5, m = 5, n = 5, temp1 = 10, temp2 = 15;
	float*** cuboid, *** resL, *** resP;
	if (argc > 1) {
		k = atoi(argv[1]), m = atoi(argv[2]), n = atoi(argv[3]),
			temp1 = atoi(argv[4]), temp2 = atoi(argv[5]);
	}
	// Linear
	cuboid = initCuboid(k, m, n);
	fillCuboid(cuboid, k, m, n, temp1, temp2);
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
	deleteCuboid(cuboid, k, m, n);
	deleteCuboid(resL, k, m, n);
	deleteCuboid(resP, k, m, n);
	filledFile.close();
	resLFile.close(); 
	resPFile.close();
	return 0;
}

// barrier(CLK_LOCAL_MEM_FENCE)
// cmd param - k, m, n, t1, t2