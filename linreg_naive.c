#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000
#define D 3
#define LR 0.01f
#define EPOCHS 10000
#define RUNS 10


double seconds_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

double min_double(double a, double b)
{
    return a<b ? a:b;
}

double max_double(double a, double b)
{
    return a>b ? a:b;
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


    double total_elapsed = 0.0;
    double best_elapsed = 1e9;
    double worst_elapsed = 0.0;

    float final_w[D] = {0.0f};
    float final_b = 0.0f;
    float final_mse = 0.0f;

    for (int run = 0; run < RUNS; run ++)
    {
        float w[D] = {0.0f};
        float b = 0.0f;
        float mse = 0.0f;
        
        double start = seconds_now();
        for (int e = 0; e < EPOCHS; e ++)
        {
            float grad_w[D];
            float grad_b;

            mse = compute_loss_and_gradients(X,y,w,b,grad_w,&grad_b);
            update_params(w,&b,grad_w,grad_b);
        }
        double end = seconds_now();
        double elapsed = end - start;
        total_elapsed += elapsed;

        best_elapsed = min_double(elapsed,best_elapsed);
        worst_elapsed = max_double(elapsed,worst_elapsed);

        for (int i = 0; i< D; i++)
        {
            final_w[i] = w[i];
        }
        final_b = b;
        final_mse = mse;
    }

    double avg_elapsed = total_elapsed/RUNS;

    printf("\nUpdated Weights/Params after %d epochs:\n", EPOCHS);

    for (int j =0; j<D; j++)
    {
        printf("w[%d] = %.8f\n", j, final_w[j]);
    }
    printf("b = %.8f\n",final_b);
    printf("mse = %.8f\n", final_mse);

    printf("\nBenchmark over %d runs:\n", RUNS);
    printf("avg_trainig_time_seconds = %.8f\n", avg_elapsed);
    printf("best_trainig_time_seconds = %.8f\n", best_elapsed);
    printf("worst_trainig_time_seconds = %.8f\n", worst_elapsed);
    printf("epochs = %d\n", EPOCHS);
    printf("samples = %d\n", N);
    printf("features = %d\n", D);
    printf("avg_time_per_epoch_ms = %.8f\n", (avg_elapsed/EPOCHS)*1000.0);
    printf("avg_samples_per_second = %.8f\n", ((double) N * EPOCHS) / avg_elapsed);
    
    return 0;
}