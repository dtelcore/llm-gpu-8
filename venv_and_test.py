import os
import subprocess
import sys

def main():
    """
    Automates the process of changing to the project directory and running 
    the workspace test script using the virtual environment's Python executable.
    """
    target_dir = r"c:\dev\llm gpu 8"
    
    # Verify the target directory exists
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)
        
    # Change current working directory to the target
    os.chdir(target_dir)
    print(f"Changed directory to: {os.getcwd()}")

    # Construct paths relative to the new current working directory
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    target_script = os.path.join("setup", "2_test_workspace.py")
    
    # Verify the virtual environment Python exists
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment Python executable not found at '{venv_python}'.")
        print("Please ensure the 'venv' has been created.")
        sys.exit(1)

    # Verify the target script exists
    if not os.path.exists(target_script):
        print(f"Error: Target script not found at '{target_script}'.")
        sys.exit(1)

    print(f"Running '{target_script}' using the virtual environment...")
    
    # Use subprocess to run the script. 
    # Calling the venv's python.exe directly bypasses the need to manually "activate" it via a shell script.
    try:
        subprocess.run([venv_python, target_script], check=True)
        print("\nSuccess: Workspace test completed.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script execution failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()