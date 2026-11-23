from itertools import product
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
import nn_bowl

@hydra.main(version_base=None, config_name='config_breadth_depth_time', config_path=nn_bowl.paths.CONFIG_PATH)
def main(cfg):
    experiment_name = cfg.experiment.name
    dir = Path(__file__).parent / experiment_name
    dir.mkdir(parents=True, exist_ok=True)
    
    # command-line overrides (except cluster.*) are forwarded to every job, e.g.
    # analysis.pruning_accuracy_threshold=0.95; they are also added to the job file names
    forwarded = [o for o in HydraConfig.get().overrides.task if not o.startswith("cluster.")]
    tag = "".join("_" + o.split("=", 1)[0].split(".")[-1] + "=" + o.split("=", 1)[1] for o in forwarded)
    job_name = f"{experiment_name}_{cfg.cluster.script}{tag}"

    # create txt file with all the jobs
    jobs_file = dir / f"cluster_jobs_{job_name}.txt"
    instances = range(cfg.experiment.n_instances)

    variables = list(cfg.experiment.variables)
    # checkpoint_elements = [var.key for var in variables]
    product_list = [instances] + [list(var.vals) for var in variables]
    combinations = list(product(*product_list))

    with open(jobs_file, "w") as f:
        for combo in combinations:
            instance = combo[0]
            var_values = combo[1:]

            cmd = f"python scripts/{cfg.cluster.script}.py"
            cmd += f" --config-name={HydraConfig.get().job.config_name}"
            cmd += f" train.instance={instance}"
            for var, value in zip(variables, var_values):
                cmd += f" {var.key}={value}"
            for o in forwarded:
                cmd += f" {o}"

            # checkpoint elements added automatically now from the variables
            # cmd += f" \"experiment.checkpoint_elements=[{','.join(checkpoint_elements)}]\""
            print(cmd)
            f.write(cmd + "\n")
    
    print("jobs: ", len(combinations))
    
    # create the slurm sbatch script
    sbatch_script_path = dir / f"cluster_sbatch_{job_name}.sh"
    script_content = f"""#!/usr/bin/bash
#
# Job script for training the models
# 
#SBATCH --job-name=nn_bowl_{cfg.cluster.script}
#SBATCH -c {cfg.train.num_workers}
#SBATCH --array=0-{len(combinations)-1}
#SBATCH --time={cfg.cluster.hours}:{cfg.cluster.minutes}:00
#SBATCH --mem-per-cpu=1gb
#SBATCH --gres=gpu:1
#SBATCH --output=batch_out/%A_%a.out
# #SBATCH --exclude=<node1>,<node2>  # adjust for your cluster

echo "=== Job started at $(date) ==="
echo "Running on node: $SLURMD_NODENAME"

echo "Activating the conda environment"
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate nn-bowl

# Per-task inductor cache so corrupt entries from killed jobs can't poison new ones
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${{USER}}_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
export TRITON_CACHE_DIR="/tmp/triton_${{USER}}_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
trap 'rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"' EXIT

# absolute path: SLURM runs a copy of this script from /var/spool/slurmd, so paths relative to it break
jobs_file="{jobs_file.resolve()}"
line_number=$((SLURM_ARRAY_TASK_ID + 1))
CMD=$(sed -n "${{line_number}}p" $jobs_file)

echo "Running command: $CMD"
eval $CMD

echo "=== Job completed at $(date) ==="
"""
    
    with open(sbatch_script_path, "w") as f:
        f.write(script_content)
    
    print(f"SBATCH script created at: {sbatch_script_path}")

if __name__ == "__main__":
    main()
