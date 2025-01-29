import subprocess

# Define parameters
params = {
    "--prob_type": "ALEPH",
    "--opt": "adam",
    "--n_models": 10,
    "--lr": 1e-5, #1e-3 for default
    "--beta1": 0.997, #0.9 for default
    "--beta2": 0.998, #0.999 for default
    "--patience": 10,
    "--save": "true",
    "--name_of_run": "adam_results"
}

# Convert parameters into a list of arguments
args = ["python", "run_trainer.py"] + [str(item) for pair in params.items() for item in pair]

# Run the command
subprocess.run(args)
