### AMOPSO: A modified particle swarm optimizer (AMPSO) for multi-modal multi-objective optimization (MMO) problems

##### Reference: Zhang X W, Liu H, Tu L P. A modified particle swarm optimization for multimodal multi-objective optimization[J]. Engineering Applications of Artificial Intelligence, 2020, 95: 103905.

##### The AMOPSO belongs to the category of multi-objective evolutionary algorithms (MOEAs). AMOPSO is an algorithm to solve the multi-modal multi-objective optimization (MMO) problems.

| Variables | Meaning                               |
| --------- | ------------------------------------- |
| npop      | Population size                       |
| iter      | Iteration number                      |
| lb        | Lower bound                           |
| ub        | Upper bound                           |
| omega     | Inertia weight                        |
| c1        | Acceleration constant 1 (default = 2) |
| c2        | Acceleration constant 2 (default = 2) |
| dim       | Dimension                             |
| pos       | Position                              |
| vmin      | Minimum velocity                      |
| vmax      | Maximum velocity                      |
| vel       | Velocity                              |
| objs      | Objectives                            |
| nvar      | The dimension of decision space       |
| nobj      | The dimension of objective space      |
| PA        | Personal archive                      |
| NA        | Neighborhood archive                  |
| PA_objs   | The objectives of PA                  |
| NA_objs   | The objective of NA                   |
| off       | Offspring                             |
| off_objs  | The objectives of offsprings          |
| scd       | Special crowding distance             |
| pf        | Pareto front                          |
| ps        | Pareto set                            |

#### Test problem: MMF1



$$
\left\{
\begin{aligned}
&f_1(x)=|x_1-2|\\
&f_2(x)=1-\sqrt{|x_1 - 2|}+2(x_2-\sin{(6 \pi |x_1 - 2| + \pi)})^2\\
&1 \leq x_1 \leq 3, -1 \leq x_2 \leq 1
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    t_npop = 800
    t_iter = 100
    t_lb = np.array([1, -1])
    t_ub = np.array([3, 1])
    main(t_npop, t_iter, t_lb, t_ub)
```

##### Output:



![](https://github.com/Xavier-MaYiMing/AMPSO/blob/main/Pareto%20front.png)

![](https://github.com/Xavier-MaYiMing/AMPSO/blob/main/Pareto%20set.png)

```python
Iteration 10 completed.
Iteration 20 completed.
Iteration 30 completed.
Iteration 40 completed.
Iteration 50 completed.
Iteration 60 completed.
Iteration 70 completed.
Iteration 80 completed.
Iteration 90 completed.
Iteration 100 completed.
```

