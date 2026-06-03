import time;
import torch;

# cant really do apples to apples since naive python so slow
N = 100000
D = 100
LR = 0.01
EPOCHS = 5000
RUNS = 10

def seconds_now():
    return time.perf_counter()


def main():
    torch.manual_seed(0)

    # for fair comp w BLAS/C
    torch.set_num_threads(1)

    print("baremetal linear regression implementation - pytorch")
    print(f"N={N}, D = {D}")

    true_w = torch.empty(D,1).uniform_(-1.0,1.0)
    true_b = torch.tensor(4.0)

    X = torch.empty(N,D).uniform_(-1.0,1.0)
    y = X @ true_w + true_b

    total_elapsed = 0.0
    best_elapsed = float('inf')
    worst_elapsed = float('-inf')

    final_w = None
    final_b = None
    final_mse = 0.0

    for i in range(RUNS):
        model = torch.nn.Linear(D,1)
        
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        
        optimizer = torch.optim.SGD(model.parameters(),lr=LR)

        start = seconds_now()
        for e in range(EPOCHS):
            pred = model(X)
            loss = torch.mean((pred-y)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = seconds_now()
        elapsed = end - start
        total_elapsed += elapsed

        best_elapsed = min(elapsed,best_elapsed)
        worst_elapsed = max(elapsed,worst_elapsed)

        final_b = model.bias.detach().clone()
        final_mse = loss.item()
        final_w = model.weight.detach().clone()

    avg_elapsed = total_elapsed/RUNS;
    print(f"\nUpdated Weights/Params after {EPOCHS} epochs:\n")
    for i in range(D):
        print(f"w[{i}] = {final_w[0, i].item():.8f}")

    print(f"b = {final_b.item():.8f}")
    print(f"mse = {final_mse:.8f}")

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