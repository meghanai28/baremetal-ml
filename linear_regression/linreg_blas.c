#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#define N 100000
#define D 100
#define LR 0.01f
#define EPOCHS 5000
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

float compute_loss_and_gradients(const float *X, const float *y, const float w[D], float b, float *pred, float grad_w[D], float *grad_b)
{

    // pred = X*w
    // y = alpha * A * x + beta * y
    // pred = 1.0 * X * w + 0.0 * pred
    cblas_sgemv(
        CblasRowMajor, // X is stored row by row
        CblasNoTrans, // use X not tranpsose
        N, //rows of X
        D, // columbs of X
        1.0f, // alpha 
        X, // matrix X
        D, // row length
        w, // vector w
        1, // step through w by 1
        0.0f, //beta
        pred, // output vector
        1 // step through predby 1
    );

    *grad_b = 0.0f;
    float total_loss = 0.0f;

    for (int i = 0; i < N; i++)
    {
        pred[i] = pred[i] + b - y[i];
        total_loss += pred[i] * pred[i];
        *grad_b += pred[i];
    }

    //grad_w = X^T * error
    cblas_sgemv(
        CblasRowMajor,
        CblasTrans,
        N,
        D,
        1.0f,
        X,
        D,
        pred,
        1,
        0.0f,
        grad_w,
        1
    );
    
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

    float true_w[D];
    float true_b = 4.0f;

    for (int j =0; j<D;j++)
    {
        true_w[j] = rand_float();
    }

    // X - features/inputs
    // y - the true label
    // BLAS - flat contigious array
    float *X = malloc(N * D * sizeof(float));
    float *y = malloc(N * sizeof(float));
    float *pred = malloc(N * sizeof(float));

    if (X == NULL || y == NULL || pred == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(X);
        free(y);
        free(pred);
        return 1;
    }

    // init w random values
    for (int i = 0; i< N; i++)
    {
        y[i] = true_b;
        for (int j = 0; j<D; j++)
        {
            X[i*D+j] = rand_float();
            y[i] += X[i * D + j] * true_w[j];
        }
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

            mse = compute_loss_and_gradients(X,y,w,b,pred,grad_w,&grad_b);
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

    printf("\nTrue Weights/Params:\n");
    for (int j = 0; j < D; j++)
    {
        printf("true_w[%d] = %.8f\n", j, true_w[j]);
    }
    printf("true_b = %.8f\n", true_b);

    printf("\nBenchmark over %d runs:\n", RUNS);
    printf("avg_training_time_seconds = %.8f\n", avg_elapsed);
    printf("best_training_time_seconds = %.8f\n", best_elapsed);
    printf("worst_training_time_seconds = %.8f\n", worst_elapsed);
    printf("epochs = %d\n", EPOCHS);
    printf("samples = %d\n", N);
    printf("features = %d\n", D);
    printf("avg_time_per_epoch_ms = %.8f\n", (avg_elapsed/EPOCHS)*1000.0);
    printf("avg_samples_per_second = %.8f\n", ((double) N * EPOCHS) / avg_elapsed);
    
    free(pred);
    free(X);
    free(y);
    return 0;
}