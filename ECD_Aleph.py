import subprocess

# Define parameters
params = {
    "--prob_type": "ALEPH",
    "--opt": "ECD_q1",
    "--n_models": 10,
    "--lr": 0.1,
    "--nu": 0,
    "--eta": 100000,
    "--F0": -0.35,
    "--loss_fn": "exp_mlc",
    "--patience": 10,
    "--save": "true",
    "--name_of_run": "ECD_results"
}

# Convert parameters into a list of arguments
args = ["python", "run_trainer.py"] + [str(item) for pair in params.items() for item in pair]

# Run the command
subprocess.run(args)
