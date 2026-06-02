#include <stdio.h>
#include <stdlib.h>

#define N 10
#define D 3
#define LR 0.01f
#define EPOCHS 5000


float rand_float(void)
{
    return ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main (void)
{
    srand(0);
    printf("baremetal linear regression implemention\n");
    printf("N = %d, D = %d\n", N, D);

    // X - features/inputs
    // y - the true label

    float X[N][D];
    float y[N];

    // init w random values
    // true weights = [3,-2,0.5]
    // true bias = 4
    for (int i = 0; i< N; i++)
    {
        for (int j = 0; j<D; j++)
        {
            X[i][j] = rand_float();
        }
        y[i] = 3.0f * X[i][0] - 2.0f * X[i][1] + 0.5f * X[i][2] + 4.0f;
    }

    // print the dataset
    for (int i = 0; i<N; i++)
    {
        printf("row %d: ", i);
        for (int j = 0; j< D; j++)
        {
            printf("X[%d][%d] = %.3f ", i,j, X[i][j]);
        }
        printf("y = %.3f\n", y[i]);
    }

    // initalize weights and bias
    float w[D] = {0.0f};
    float b = 0.0f;

    printf("\n inital model params \n");

    for (int i = 0; i<D; i ++)
    {
        printf("w[%d] = %.3f\n", i, w[i]);
    }
    printf("b = %.3f\n", b);


    for (int e = 0; e < EPOCHS; e ++)
    {
        float grad_w[D] = {0.0f};
        float grad_b = 0.0f;
        float total_loss = 0.0f;

        for (int i = 0; i < N; i++)
        {
            float pred = b;
            for (int j = 0; j<D; j++)
            {
                pred += X[i][j] * w[j];
            }
            float r_error = pred - y[i];

            for (int j = 0; j<D; j++)
            {
                grad_w[j] += r_error * X[i][j];
            }

            total_loss += r_error * r_error;
            grad_b += r_error;
        }

        for (int i = 0; i <D; i++)
        {
            grad_w[i] = (2.0f/N) * grad_w[i];
            w[i] -= LR * grad_w[i];
        }
        
        grad_b = (2.0f/N) * grad_b;
        b -= LR * grad_b;
        
        float mse = total_loss/N;

        if (e % 100 == 0)
        {
            printf("epoch = %d & MSE = %.3f\n", e, mse);
        }
    }

    printf("\n Updated Weights/Params after %d epochs:\n", EPOCHS);

    for (int j =0; j<D; j++)
    {
        printf("w[%d] = %.3f\n", j, w[j]);
    }
    printf("b = %.3f\n",b);

    return 0;

}