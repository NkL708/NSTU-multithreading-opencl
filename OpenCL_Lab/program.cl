__kernel void test()
{
	int id = get_local_id(0);
	printf("Hello World from host! id - %d\n", id);
}

// barrier(CLK_LOCAL_MEM_FENCE)