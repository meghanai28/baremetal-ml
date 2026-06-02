#include <stdio.h>
#include <stdlib.h>

#define N 1000
#define D 3
#define LR 0.01f
#define EPOCHS 2000


float rand_float(void)
{
    return ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
}

float predict(const float x[D], const float w[D], float b)
{
    float pred = b;
    for (int j = 0; j<D; j++)
    {
        pred += x[j] * w[j];
    }
    return pred;
}

float compute_loss_and_gradients(const float X[N][D], const float y[N], const float w[D], float b, float grad_w[D], float *grad_b)
{

    for (int j = 0; j < D; j++)
    {
        grad_w[j] = 0.0f;
    }
    *grad_b = 0.0f;
    float total_loss = 0.0f;

    for (int i = 0; i < N; i++)
    {
        float pred = predict(X[i],w,b);
        float r_error = pred - y[i];

        for (int j = 0; j<D; j++)
        {
            grad_w[j] += r_error * X[i][j];
        }

        total_loss += r_error * r_error;
        *grad_b += r_error;
    }

    for (int i = 0; i <D; i++)
    {
        grad_w[i] = (2.0f/N) * grad_w[i];
    }
        
    *grad_b = (2.0f/N) * (*grad_b);
        
    float mse = total_loss/N;
    return mse;
}

void update_params(float w[D], float *b, const float grad_w[D], float grad_b)
{
    for (int i =0; i<D; i++)
    {
        w[i] -= LR * grad_w[i];
    }
    *b -= LR * grad_b;
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

    // initalize weights and bias
    float w[D] = {0.0f};
    float b = 0.0f;

    for (int e = 0; e < EPOCHS; e ++)
    {
        
        float grad_w[D];
        float grad_b;

        float mse = compute_loss_and_gradients(X,y,w,b,grad_w,&grad_b);
        update_params(w,&b,grad_w,grad_b);

        if (e % 100 == 0)
        {
            printf("epoch = %d & MSE = %.8f\n", e, mse);
        }
    }

    printf("\nUpdated Weights/Params after %d epochs:\n", EPOCHS);

    for (int j =0; j<D; j++)
    {
        printf("w[%d] = %.3f\n", j, w[j]);
    }
    printf("b = %.3f\n",b);

    return 0;
}