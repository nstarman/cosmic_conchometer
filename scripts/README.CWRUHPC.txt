# Getting onto a Node

Interactive node:

```bash
>>> srun -N 1 -n 1 -c 2 --time=2:00:00 --mem=5gb --pty /bin/bash

```

- - -

module swap intel gcc
module load python/3.8.6
source ~/venv/cosmo/bin/activate
