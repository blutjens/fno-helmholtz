import ray
from tqdm import tqdm
import psutil

"""
Parallelization
"""
def to_iterator(obj_id):
    # Call this to display tqdm progressbar when using ray parallel processing
    # Source https://github.com/ray-project/ray/issues/5554
    while obj_id:
        done, obj_id = ray.wait(obj_id)
        yield ray.get(done[0])

def init_preprocessing(fn, parallel=False):
    """
    Init parallel processing
    Source: https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8

    Args:
        fn (fn): Function that's to be parallelized

    Returns:
    """
    if parallel:
        num_cpus = psutil.cpu_count(logical=False)
        print('n cpus', num_cpus)
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)            

    if parallel:
        fn_r = ray.remote(fn).remote
    else:
        fn_r = fn

    fn_tasks = []
    return fn_r, fn_tasks

def get_parallel_fn(model_tasks):
    """
    Waits for parallel model tasks to finish and returns outputs
    
    Args:
        model_outputs (list(tuple)): Outputs of model
    """
    for x in tqdm(to_iterator(model_tasks), total=len(model_tasks)):
        pass
    model_outputs = ray.get(model_tasks) # [0, 1, 2, 3]
    return model_outputs
