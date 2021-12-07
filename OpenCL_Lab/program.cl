__kernel void distributeKernel(__global float* cuboid, int k, int m, int n, __global float* result)
{
	int index = get_global_id(0);
	result[index] = cuboid[index];
	bool isDissipated = false;
	// Ends if temperatures in cube becomes balanced
	while (!isDissipated) {
		// Calc average temperature
		int z, y, x;
		float sum = 0;
		int count = 0;
		float average;
		z = index / (k * m);
		y = (index % (k * m)) / n;
		x = index % m;
		for (int zSum = z - 1; zSum <= z + 1; zSum++) {
			for (int ySum = y - 1; ySum <= y + 1; ySum++) {
				for (int xSum = x - 1; xSum <= x + 1; xSum++) {
					if (zSum >= 0 && ySum >= 0 && xSum >= 0
						&& zSum < k && ySum < m && xSum < n) {
						count++;
						sum += result[(zSum * k * m) + (ySum * m) + xSum];
					}
				}
			}
		}
		average = round(sum / count * 100) / 100;
		//result[index]
		if (average == result[index]) {
			isDissipated = true;
		}
		else {
			result[index] = average;
		}
	}
}