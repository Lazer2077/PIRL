# Physics-Informed Model Based Reinforcement Learning

## Environment
Install the mujuco-gym

```bash
pip install gymnasium[mujoco] mujoco
```

Intall the Requirements

```bash
pip install -r requirements.txt
```

# Run the code
Select the method in the [OptMethods/__init__.py](OptMethods/__init__.py) and enviroment in the [gymnasium](https://gymnasium.farama.org/content/environment_creation/) to run the code.

Available gymnasium enviroments:

- `cartpole-v1`
- `Acrobot-v1`
- `Pendulum-v1`
- `HalfCheetah-v4`
- `Ant-v4`




```bash
python mainRunRL.py
```


## Log and Plot
Run the `plot.py` to plot the training reward and loss. 

```bash
python plot.py
```

The log will be save in the `LogTmp` folder, with the name format as `MM_DD_HH_MM_Method_EnvName`.

If use `ploty`, we can view some plots in the browser.
The plot will be save in the project folder, with the name format as `Training Curve.html`.
If not use `ploty`, `matplotlib` will be used to plot the training curve, and the plot will be save in the project folder, with the name format as `Training Curve.png`.





