// compute vector sum h_c = h_a + h_b

#define N 10

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    for (int i = 0; i < n; i++) h_C[i] = h_A[i] + h_B[i];
}

int main()
{
    float *h_A, *h_B, *h_C;

    vecAdd(h_A, h_B, h_C, N);
}
