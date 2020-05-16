import json
from Neuirps_BEL2SCM.extract_nodes import get_nodes
from Neuirps_BEL2SCM.causal_model import *


def main():
    # config = json.load(open("config.json"))
    f = open('config.json')
    config = json.load(f)

    file_type = config["bel_settings"]["file_type"]
    file_name = config["bel_settings"]["file_name"]
    nodes = get_nodes(file_type=file_type, file_name=file_name)
    # print(nodes)
    print(SCM(nodes, config))

if __name__ == "__main__":
    main()