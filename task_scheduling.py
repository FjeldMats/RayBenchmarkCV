import ray
from ray.cluster_utils import Cluster
from ray.util.queue import Queue
import time, warnings, logging
from datetime import datetime

def start_cluster(num_cpus):
    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": num_cpus})
    warnings.filterwarnings('ignore')
    ray.init(address=cluster.address, logging_level=logging.ERROR)
    assert ray.cluster_resources()["CPU"] == num_cpus

#simulates a long running task
@ray.remote(num_cpus=1)
def run_ml_task(t, theta):
    t = t * theta
    #print(f"Starting task {rnd}")
    time.sleep(t)
    return t


def get_runtime():
    while True:
        yield 3

def correctness_check(lst): 
    return sum(lst)

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

@ray.remote
def _rayNestedCVSelection(outer, inner, param, num_cpus=1):
    setting_results = [] 

    tasks = []
 
    print(f"testing: {param}")
    for outer_index in range(outer):
        for inner_index in range(inner):
            tasks.append(run_ml_task.remote(next(get_runtime()), param))
    
    for _ in range(len(tasks)//num_cpus):
        setting_results.extend(ray.get(tasks[:num_cpus]))
        tasks = tasks[num_cpus:]
        #print(f"batched {len(setting_results)} tasks")
    
    if len(tasks) > 0:
        setting_results.extend(ray.get(tasks))
    
    return sorted(setting_results)[0], setting_results 


def rayNestedCVSelection(outer, inner, generator):
    total = []
    best_settings = []
    hyperparam = next(generator)

    total_cpus = int(ray.cluster_resources()["CPU"])

    while hyperparam is not None: 

        setting_results = []
        
        num_cpus = total_cpus // len(hyperparam) 
        if num_cpus == 0:
            num_cpus = 1

        print(f"using {num_cpus} cpus per setting")
        for param in hyperparam:
            setting_results.append(_rayNestedCVSelection.remote(outer, inner, param, num_cpus))
                
        results = ray.get(setting_results)
        for b, t in results:
            total.extend(t)
            best_settings.append(b)

        hyperparam = next(generator) # input results to hyperparam generator
    return sorted(best_settings)[0], total

def run_benchmark(outer, inner, hyperparams):

    execution_times = []
    work = []
    

    print("Nested cross validation with Ray parallel selection (Grid search case)")
    # Nested cross validation with Ray
    start_time = time.perf_counter()
    res, lst = rayNestedCVSelection(outer, inner, paralell_hyperparam(hyperparams, hyperparams)) # grid searach 
    print(res)
    print(correctness_check(lst))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')
    

    print("Nested cross validation with Ray parallel selection (Bayesian optimization case)")
    # Nested cross validation with Ray
    start_time = time.perf_counter()
    res, lst = rayNestedCVSelection(outer, inner, paralell_hyperparam(hyperparams, 2)) # bayesian optimization
    print(res)
    print(correctness_check(lst))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    execution_times.append(total_time)
    print(f'{total_time:.4f} seconds\n')

    return execution_times

cpus = list(reversed([64]))
splits = [(5, 5)]
hyperparams = [5]

methods = ["rayNestedCVSelection (Grid search)", "rayNestedCVSelection (Bayesian optimization)"]

with open(f"results/cpu-{len(cpus)} splits-{len(splits)} params-{len(hyperparams)}-{datetime.now()}.csv", 'w') as f:
    f.write(f"method, cpu, split_outer, split_inner, param, time\n")
    for param in hyperparams:
        for cpu in cpus:
            for split in splits:  
                start_cluster(cpu) # start cluster
                print(f"Running with {cpu} cpus and {split} splits")
                result = run_benchmark(split[0], split[1], param)
                print(result)
                for method in methods:
                    f.write(f"{method}, {cpu},{split[0]},{split[1]},{param},{result[methods.index(method)]}\n")
                f.flush()
                ray.shutdown()
    f.close()