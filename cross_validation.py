import ray
from ray.cluster_utils import Cluster
import time
import random
import warnings
import logging
from datetime import datetime

# simulates a long running task
@ray.remote(num_cpus=1)
def run_ml_task(t, theta):
    t = t * theta
    # print(f"Starting task {rnd}")
    time.sleep(t)
    return t


def get_runtime():
    while True:
        yield 3


def hyperparam(length):
    for i in range(1, length+1):
        yield i
    else:
        yield None


def paralell_hyperparam(length, num):
    counter = 1
    ranges = []
    for i in range(length//num):
        ranges.append((counter, counter+num))
        counter += num
    if length % num != 0:
        ranges.append((counter, length+1))

    for start, end in ranges:
        yield range(start, end)
    else:
        yield None


def CV(splits, hyperparams):
    total = []
    best = []
    theta = next(hyperparams)
    gen_time = get_runtime()
    while theta is not None:
        # print(f"Starting theta: {theta}")
        tasks = [run_ml_task.remote(next(gen_time), theta)
                 for _ in range(splits)]
        results = ray.get(tasks)
        best.append(sorted(results)[0])  # get the best result
        total.extend(results)
        theta = next(hyperparams)

    return sorted(best)[0], total


def nestedCV(outer, inner, hyperparams):
    total = []
    best_settings = []
    for i in range(outer):
        print(f"Starting outer loop {i+1}/{outer}")
        best, sub_total = CV(inner, hyperparam(hyperparams))
        best_settings.append(best)
        total.extend(sub_total)

    return sorted(best_settings)[0], total


@ray.remote
def _nestedCV(inner, hyperparams):
    print(f"Starting outer loop")
    return CV(inner, hyperparam(hyperparams))


def rayNestedCV(outer, inner, hyperparams):
    total = []
    best_settings = []
    tasks = [_nestedCV.remote(inner, hyperparams) for _ in range(outer)]
    results = ray.get(tasks)

    for b, t in results:
        total.extend(t)
        best_settings.append(b)

    return sorted(best_settings)[0], total


def rayNestedCVUnnested(outer, inner, hyperparams):
    total = []

    best_settings = []
    generator = hyperparam(hyperparams)
    param = next(generator)
    while param is not None:

        setting_results = []

        print(f"testing: {param}")
        for outer_index in range(outer):
            for inner_index in range(inner):
                setting_results.append(
                    run_ml_task.remote(next(get_runtime()), param))

        results = ray.get(setting_results)
        total.extend(results)
        best_settings.append(sorted(results)[0])
        param = next(generator)  # input results to hyperparam generator
    return sorted(best_settings)[0], total


@ray.remote
def _rayNestedCVSelection(outer, inner, param):
    setting_results = []

    print(f"testing: {param}")
    for outer_index in range(outer):
        for inner_index in range(inner):
            setting_results.append(
                run_ml_task.remote(next(get_runtime()), param))

    setting_results = ray.get(setting_results)
    return sorted(setting_results)[0], setting_results


def rayNestedCVSelection(outer, inner, generator):
    total = []
    best_settings = []
    hyperparam = next(generator)
    while hyperparam is not None:

        print(f"running: {hyperparam}")
        setting_results = []
        for param in hyperparam:
            # print(f"param: {param}")
            setting_results.append(
                _rayNestedCVSelection.remote(outer, inner, param))

        results = ray.get(setting_results)
        for b, t in results:
            total.extend(t)
            best_settings.append(b)

        hyperparam = next(generator)  # input results to hyperparam generator
    return sorted(best_settings)[0], total


def correctness_check(lst):
    return sum(lst)


def run_benchmark(outer, inner, hyperparams):

    execution_times = []
    work = []

    # Cross validation
    print("Cross validation")
    start_time = time.perf_counter()
    res = CV(outer, hyperparam(hyperparams))
    print(res)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')

    random.seed(11)

    # Nested cross validation
    print("Nested cross validation")
    start_time = time.perf_counter()
    res = nestedCV(outer, inner, hyperparams)
    print(res)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')

    random.seed(11)
    print("Nested cross validation with Ray")
    # Nested cross validation with Ray
    start_time = time.perf_counter()
    res, lst = rayNestedCV(outer, inner, hyperparams)
    print(res)
    print(correctness_check(lst))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')

    random.seed(11)
    print("Nested cross validation with Ray unnested")
    # Nested cross validation with Ray
    start_time = time.perf_counter()
    res, lst = rayNestedCVUnnested(outer, inner, hyperparams)
    print(res)
    print(correctness_check(lst))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')

    random.seed(11)
    print("Nested cross validation with Ray parallel selection (Grid search case)")
    # Nested cross validation with Ray
    start_time = time.perf_counter()
    res, lst = rayNestedCVSelection(outer, inner, paralell_hyperparam(
        hyperparams, hyperparams))  # grid searach
    print(res)
    print(correctness_check(lst))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')

    random.seed(11)
    print("Nested cross validation with Ray parallel selection (Bayesian optimization case)")
    # Nested cross validation with Ray
    start_time = time.perf_counter()
    res, lst = rayNestedCVSelection(outer, inner, paralell_hyperparam(
        hyperparams, 2))  # bayesian optimization
    print(res)
    print(correctness_check(lst))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')

    return execution_times


def start_cluster(num_cpus):
    cluster = Cluster(initialize_head=True, head_node_args={
                      "num_cpus": num_cpus})
    warnings.filterwarnings('ignore')
    ray.init(address=cluster.address, logging_level=logging.ERROR)
    assert ray.cluster_resources()["CPU"] == num_cpus


cpus = list(reversed([256]))
splits = [(5, 5)]
hyperparams = [5]


methods = ["CV", "Nested CV", "rayNestedCV", "rayNestedCVUnnested",
           "rayNestedCVSelection (Grid search)", "rayNestedCVSelection (Bayesian optimization)"]


if __name__ == "__main__":

    print(
        f"running: {len(list(cpus)) * len(splits) * len(hyperparams)} experiments")

    with open(f"results/cpu-{len(cpus)} splits-{len(splits)} params-{len(hyperparams)}-{datetime.now()}.csv", 'w') as f:
        f.write(f"method, cpu, split_outer, split_inner, param, time\n")
        for param in hyperparams:
            for cpu in cpus:
                for split in splits:
                    start_cluster(cpu)  # start cluster
                    print(f"Running with {cpu} cpus and {split} splits")
                    result = run_benchmark(split[0], split[1], param)
                    print(result)
                    for method in methods:
                        f.write(
                            f"{method}, {cpu},{split[0]},{split[1]},{param},{result[methods.index(method)]}\n")
                    f.flush()
                    ray.shutdown()
        f.close()
