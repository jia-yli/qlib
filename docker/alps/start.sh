# build
srun -p debug --cpus-per-task 128 --mem 384G -N 1 --pty --account a-g34 bash
srun -p debug --cpus-per-task 128 --mem 384G -N 1 --gpus 4 --pty --account a-g34 bash
srun --reservation interact --cpus-per-task 128 --mem 384G -N 1 --gpus 4 --pty --account a-g34 bash
# env, cpu-only
srun -p debug --cpus-per-task 128 --mem 384G -N 1 --environment=/users/ljiayong/projects/qlib/docker/alps/eiger.toml --pty --account g34 bash
srun -p debug --cpus-per-task 128 --mem 384G -N 1 --gpus 4 --environment=/users/ljiayong/projects/qlib/docker/alps/clariden.toml --pty --account a-g34 bash
srun --reservation interact --cpus-per-task 128 --mem 384G -N 1 --gpus 4 --environment=/users/ljiayong/projects/qlib/docker/alps/clariden.toml --pty --account a-g34 bash