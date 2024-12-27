import hydra
from hydra.core.plugins import Plugins
from hydra.types import RunMode

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(_cfg):
    print("Plugins detected by Hydra:")
    for plugin in Plugins.discover():
        print(f"  - {plugin.__name__}")

if __name__ == "__main__":
    main()