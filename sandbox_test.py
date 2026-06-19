import arcengine
import os

# This prints the exact folder where the arcengine package is installed
engine_path = arcengine.__path__[0]
print("ARC Engine is installed at:", engine_path)

# You can list the files inside it to see where the games are bundled
print("Contents:", os.listdir(engine_path))
