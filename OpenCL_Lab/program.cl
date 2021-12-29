__kernel void distributeKernel(__global float* cuboid, int k, int m, int n, __global float* result)
{
	// ThreadIndex - row y
	int threadIndex = get_global_id(0);
	int numOfThreads = k * m;
	// Copy
	// (z * m * n) + (y * n) + x
	int z = threadIndex / m;
	int y = threadIndex % m;
	if (threadIndex == 0) {
		printf("cuboid = %d\n", result[0]);
		printf("cuboid = %d\n", result[1]);
		printf("cuboid = %d\n", result[2]);
		printf("cuboid = %d\n", result[4]);
		printf("cuboid = %d\n", result[5]);
		printf("cuboid = %d\n", result[6]);
		printf("cuboid = %d\n", result[7]);
	}
	for (int i = (z * m * n) + (y * n); i < (i + n - 1); i++) {
		result[i] = cuboid[i];
	}
	bool isDissipated = false;
	int dissipatedCount = 0;
	// Ends if temperatures in cube becomes balanced

}
/*
	while (!isDissipated) {
		// Calc average temperature
		float sum = 0;
		int count = 0;
		float average;
		int z = threadIndex / m;
		int y = threadIndex / k;
		for (int x = 0; x < n; x++) {
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
		}
		average = round(sum / count * 100) / 100;
		//result[index]
		if (average == result[threadIndex]) {
			dissipatedCount++;
		}
		if (dissipatedCount == n) {
			isDissipated = true;
		}
		else {
			result[threadIndex] = average;
		}
	}
*/