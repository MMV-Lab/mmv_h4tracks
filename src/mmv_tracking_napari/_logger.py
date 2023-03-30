import sys

def setup_logging():
    plugin_directory = __file__.removesuffix('/src/mmv_tracking_napari/_logger.py')
    path = "{}/hitl4trk.log".format(plugin_directory)
    sys.stdout = open(path, 'w')
    print("logging initialized")
        
def log(text, path = "hitl4trk.log"):
    with open(path, 'w') as file:
        file.write(text)