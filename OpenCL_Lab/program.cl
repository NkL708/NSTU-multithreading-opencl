__kernel void distributeKernel(__global float* cuboid, int k, int m, int n, __global float* result)
{
	// ThreadIndex - row y
	int threadIndex = get_global_id(0);
	int numOfThreads = k * m;
	// Copy
	// (z * m * n) + (y * n) + x
	int z = threadIndex / m;
	int y = threadIndex % m;
	for (int i = (z * m * n) + (y * n), begin = i; i < (begin + n); i++) {
		result[i] = cuboid[i];
	}
	bool isDissipated = false;
	int dissipatedCount = 0;
	// Ends if temperatures in cube becomes balanced
	while (!isDissipated) {
		// Calc average temperature
		float sum = 0;
		int count = 0;
		float average;
		for (int x = 0; x < n; x++) {
			int index = (z * m * n) + (y * n) + x;
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
			if (dissipatedCount == n) {
				isDissipated = true;
			}
		}
	}
}