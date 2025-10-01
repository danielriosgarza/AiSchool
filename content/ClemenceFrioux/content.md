# Tutorial on metabolic modelling

## Using conda

The following has to be done once

```
# load the module
module load conda

conda init
```

To activate an environment:

```
conda activate my_env
````

To deactivate an environment:

```
conda deactivate
```

## Paths to the conda environments for this tutorial

```
/shared/projects/tp_2534_ai_microbiomes_181502/conda/envs/ebame_metabo_gapseq
/shared/projects/tp_2534_ai_microbiomes_181502/conda/envs/ebame_metabo_reasoning
/shared/projects/tp_2534_ai_microbiomes_181502/conda/envs/numerical_modelling
```

We will mostly be using `/shared/projects/tp_2534_ai_microbiomes_181502/conda/envs/ebame_metabo_reasoning`

## Tutorial

See [https://gitlab.inria.fr/cfrioux/ebame](https://gitlab.inria.fr/cfrioux/ebame)
The tutorial has a section dedicated to metabolic network reconstruction which we will not consider during this training.

We will do the following:

- Manipulation of metabolic models with [Fluxer](https://fluxer.umbc.edu/)
- A closer look at a metabolic network: the SBML file
- Metabolic network modelling
    - [Using toy data](https://gitlab.inria.fr/cfrioux/ebame#using-toy-data-)
    - [Using real networks](https://gitlab.inria.fr/cfrioux/ebame#using-real-networks-)
- Screening the metabolism of microbial communities
    - [Using toy community data](https://gitlab.inria.fr/cfrioux/ebame#using-toy-community-data-)
    - [Using realistic communities](https://gitlab.inria.fr/cfrioux/ebame#using-realistic-communities-)
- BONUS - [Inferring seed metabolites (nutrients) from the metabolic network](https://gitlab.inria.fr/cfrioux/ebame#inferring-seed-metabolites-nutrients-from-the-metabolic-network)

To start, get the data by cloning the git repository:
```
git clone https://gitlab.inria.fr/cfrioux/ebame.git
```

And activate the following environment:
```
conda activate /shared/projects/tp_2534_ai_microbiomes_181502/conda/envs/ebame_metabo_reasoning
```

## Wanna play with Answer Set Programming?

```
reaction("r1").
reactant("A", "r1").
product("B", "r1").

reaction("r2").
reactant("B", "r2").
reactant("C", "r2").
product("D", "r2").

seed("A").
seed("C").

scope(M) :- seed(M).
scope(M) :- product(M,R); reaction(R); scope(N) : reactant(N,R).

#show scope/1.
```

Try it on the [clingo online solver](https://potassco.org/clingo/run/)






