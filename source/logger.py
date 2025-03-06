import logging
import os
from datetime import datetime

# Get the base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create a log file name with the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the logs directory path
logs_directory = os.path.join(BASE_DIR, "logs")

# Create the logs directory if it doesn't exist
os.makedirs(logs_directory, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(logs_directory, LOG_FILE)

# Configure logging
# Configure logging (file + console)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger().addHandler(logging.StreamHandler())  # Show logs in console
print("Logger file completed")

