#%%
import os
import mlflow
import mlflow.sklearn
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts


#%%
if __name__ == "__main__":
    ri = randint(0, 100)
    r = random()

    log_param('param1', ri)

    print(ri, r)
    log_metric("foo", r)
    log_metric("foo", r+1)
    log_metric("foo", r+2)

    path = "outputs"

    os.makedirs(path, exist_ok=True)

    with open(f"{path}/test.txt", "w") as f:
        f.write("hello world")
    
    log_artifacts(f"{path}")

