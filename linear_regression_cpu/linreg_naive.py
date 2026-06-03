import time;
import random;

# cant really do apples to apples since naive python so slow
N = 1000
D = 30
LR = 0.01
EPOCHS = 5000
RUNS = 3

def seconds_now():
    return time.perf_counter()

def rand_float():
    return random.random() * 2.0 - 1.0

# inline would make this faster 
# kept as is for now since this implementation is not that important
def predict(x,w,b):
    pred = b
    for i in range(D):
        pred += x[i] * w[i];
    return pred

def compute_loss_and_gradients(X, y, w, b):
    grad_w = [0.0 for _ in range(D)]
    grad_b = 0.0
    total_loss = 0.0

    for i in range(N):
        pred = predict(X[i],w,b)
        r_error = pred - y[i]

        for j in range(D):
            grad_w[j] += r_error * X[i][j]

        total_loss += r_error ** 2
        grad_b += r_error

    for i in range(D):
        grad_w[i] = (2.0/N) * grad_w[i]
    grad_b = (2.0/N) * grad_b
    mse = total_loss/N
    return grad_w, grad_b, mse

def update_params(w,b,grad_w, grad_b):
    for i in range(D):
        w[i] -= LR * grad_w[i]
    b -= LR * grad_b
    return b


def main():
    random.seed(0)
    print("baremetal linear regression implementation - python")
    print(f"N={N}, D = {D}")

    true_w = [rand_float() for _ in range(D)]
    true_b = 4.0

    X = [[rand_float() for _ in range(D)] for _ in range(N)]
    y = [true_b + sum(X[i][j] * true_w[j] for j in range(D)) for i in range(N)]

    total_elapsed = 0.0
    best_elapsed = float('inf')
    worst_elapsed = float('-inf')

    final_w = [0.0 for _ in range(D)]
    final_b = 0.0
    final_mse = 0.0

    for i in range(RUNS):
        w = [0.0 for _ in range(D)]
        b = 0.0
        mse = 0.0

        start = seconds_now()
        for e in range(EPOCHS):
            grad_w, grad_b, mse = compute_loss_and_gradients(X, y, w, b)
            b = update_params(w,b,grad_w,grad_b)

        end = seconds_now()
        elapsed = end - start
        total_elapsed += elapsed

        best_elapsed = min(elapsed,best_elapsed)
        worst_elapsed = max(elapsed,worst_elapsed)

        final_b = b
        final_mse = mse
        final_w = w.copy()

    avg_elapsed = total_elapsed/RUNS;
    print(f"\nUpdated Weights/Params after {EPOCHS} epochs:\n")
    for i in range(D):
        print(f"w[{i}] = {final_w[i]}\n")

    print(f"b = {final_b}\n")
    print(f"mse = {final_mse}\n")

    print(f"\nBenchmark over {RUNS} runs:")
    print(f"avg_training_time_seconds = {avg_elapsed:.8f}")
    print(f"best_training_time_seconds = {best_elapsed:.8f}")
    print(f"worst_training_time_seconds = {worst_elapsed:.8f}")
    print(f"epochs = {EPOCHS}")
    print(f"samples = {N}")
    print(f"features = {D}")
    print(f"avg_time_per_epoch_ms = {(avg_elapsed / EPOCHS) * 1000.0:.8f}")
    print(f"avg_samples_per_second = {(N * EPOCHS) / avg_elapsed:.8f}")

    return 0

if __name__ == "__main__":
    main()