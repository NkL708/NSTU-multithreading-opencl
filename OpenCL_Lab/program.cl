__kernel void distributeKernel(__global float*** cuboid, int k, int m, int n, __global float*** result)
{
	int gz = get_global_id(0);
	int gy = get_global_id(1);
	printf("gy - %d \n", &gy);
	printf("gz - %d \n", &gz);
	bool isDissipated = false;
	int size = k * m * n;
	// Ends if temperatures in cube becomes balanced
	while (!isDissipated) {
		int dissipatedCount = 0;
		for (int x = 0; x < n; x++) {
			// Calc average temperature
			float sum = 0;
			int count = 0;
			float average;
			for (int zSum = gz - 1; zSum <= gz + 1; zSum++) {
				for (int ySum = gy - 1; ySum <= gy + 1; ySum++) {
					for (int xSum = x - 1; xSum <= x + 1; xSum++) {
						if (zSum >= 0 && ySum >= 0 && xSum >= 0
							&& zSum < k && ySum < m && xSum < n) {
							count++;
							sum += result[gz][gy][xSum];
						}
					}
				}
			}
			average = round(sum / count * 100) / 100;
			if (average == result[gz][gy][x]) {
				dissipatedCount++;
			}
			else {
				result[gz][gy][x] = average;
			}
		}
		if (dissipatedCount == size) {
			isDissipated = true;
		}
	}
}