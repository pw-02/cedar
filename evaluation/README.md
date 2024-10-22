# Evaluation

The following contains instructions to setup and run evaluation to reproduce the results in our VLDB paper.

Make sure you have completed the setup in the top-level README.

### Setting up Baselines
cedar compares against multiple baselines, including PyTorch DataLoaders, tf.data, tf.data service, Ray DataSets, FastFlow, and Plumber.
The above pip installation installs PyTorch DataLoaders, and tf.data.
We provide setup instructions for the other following baselines below.

#### tf.data service
tf.data is already installed using the above commands.
To use tf.data service, you must create another VM with all dependencies and datasets already installed (as mentioned above for setting up Ray).

Prior to running tf.data service benchmarks, you must manually start up the other VM (we used an `n2-standard-32`), source your `venv`, and then run `python <PATH_TO_REPO>/evaluation/run_tf_service.py --ip_addr <IP_ADDR>` to launch the tf.data service worker process.
The process must be continuously running when you run tf.data service benchmarks on your local node (we used an `n2-standard-8`).

#### Ray Data
To use Ray Data, first create a Ray Cluster using the `ray up` command as mentioned above.
This will create a remote VM that Ray Data will use for processing.
However, to ensure that Ray can run tasks on both the remote and local VM, connect your local VM by executing the following on your local VM.

`ray start --address='10.138.0.79:6379’`

Note that the address should be the one that is provided to you after you run the `ray up` command.

Remember to run `ray stop` and `ray down -y configs/ray.yaml` when you are done.

#### FastFlow
Follow the instructions at https://github.com/SamsungLabs/FastFlow to install FastFlow.
It is recommended that you do this in a separate venv, as FastFlow must build a custom version of TensorFlow from scratch.

After installation, follow the instructions for tf.data service to clone a new VM and launch the remote worker for processing (with the same `run_tf_service.py` script).
Note that you must use the environment that has FastFlow installed when running the worker process on the remote node.

#### Plumber
Follow the instructions at https://github.com/mkuchnik/PlumberApp to install Plumber.
As with FastFlow, this requires a custom TensorFlow installation, so be sure to do this in a separate environment.

You must also install the following package in the Plumber environment.
`pip install tensorflow-addons==0.16.1`


## Running Paper Experiments

#### Baseline Comparison
Once all dependencies and datasets are installed/downloaded, you are ready to run experiments.
Note that `cedar-remote`, Ray Data, tf.data service, and FastFlow require a running remote VM (n2-standard-32), as specified in the setup instructions above.
It is recommended to run experiments for one framework at a time, since each requires different setup steps and potentially different environments.

We provide all scripts to run experiments and generate plots in `<PATH_TO_REPO>/evaluation`.
Note that these commands should be run on the local VM (n2-standard-8).

- cedar_local: `run_cedar_local.sh`
- torch: `run_torch.sh`
- ray_local: `run_ray_local.sh`
- tf: `run_tf.sh`
- plumber: `plumber/<path_to_pipeline>/run_plumber.sh`

- cedar_remote: `run_cedar_remote.sh` (note, requires Ray cluster running. You must first manually update the IP address for the ray head node in the script).
- ray remote: `run_ray_remote.sh` (note, requires Ray Data setup as detailed above)
- tf.data service: `run_tf_service.sh` (note, requires tf.data service worker setup. You must also manually update the worker IP address in the script)
- fastflow: `fastflow/examples/run_fastflow.sh` (note, requires FastFlow worker setup. You must also manually update the worker IP in the `config.yaml` file in the fastflow directory)

Each script will run all pipelines back to back. After running each pipeline, the evaluation script will report the execution time.
To generate the plots, first update the results file with the reported throughput (`aggregate_data.csv`, convert reported time and number of samples to throughput (samples/s)).

Then, run the plot script `python plot_benchmarks.py`.

NOTE: many of the above scripts require a Ray IP to be set (after you configure your ray cluster).
You may have to modify the scripts to reflect the correct IP.

#### Scaling
To generate auto-tuning results, run `run_autotuning.sh` (with the Ray cluster set up).
This will sweep the targe throughput from 40 to 600.
Looking at the cedar log (in /tmp/), this will report the number of processes that the cedar scaler settles to.
Modify the `observed_procs` field to this value and run `plot_scaling.py` to regenerate the plot.
To sweep the set_procs and observed_tput, manually generate a config for the pipeline and change the number of workers.

#### Ablation
To run ablation experiments, run `run_ablation.sh` (with the Ray cluster set up).
As with the baseline experiments, each command will run a separate job and report execution times.
To generate the ablation plot, update the corresponding reported execution times (plots/ablation.csv) and run `python plots/plot_ablation.py`.

#### Caching
To run caching experiments, run `run_cache.sh` (with the Ray cluster set up).
As with the baseline experiments, each command will run a separate job, one with caching and one without caching, for the three pipelines in Table 3.
Table 3 reports the cache throughput normalized to the throughput without caching.