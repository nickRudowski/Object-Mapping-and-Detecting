# Driver code for the 
import subprocess

def run_task(script_name, *args):
    """
    Runs a Python script as a subprocess, the goal is to not have to edit the task codes.
    Parameters:
        script_name (str): The name of the script to run.
        *args: Arguments to pass to the script.
    """
    command = ["python", script_name] + list(args)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Output from {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")


def main():
    #Execeute Task 1
    print("Starting Task 1:")
    run_task("Task1.py")
    print("Task 1 completed.")

    #Execute Task 2
    print("Starting Task 2:")
    run_task("Task2.py")
    print("Task 2 completed.")

    #Execeute Task 3
    print("Starting Task 3:")
    run_task("Task3.py")
    print("Task 3 completed.")


if __name__ == "__main__":
    main()
