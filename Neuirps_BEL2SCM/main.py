import json
# from Neuirps_BEL2SCM.extract_nodes import get_nodes
from Neuirps_BEL2SCM.SCM import SCM


def main():
    # config = json.load(open("config.json"))
    # f = open('config.json')
    # config = json.load(f)
    #
    # file_type = config["bel_settings"]["file_type"]
    # file_name = config["bel_settings"]["file_name"]
    # nodes = get_nodes(file_type=file_type, file_name=file_name)
    # # print(nodes)
    # print(SCM(nodes, config))

    # bel and config file path
    bel_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\BELSourceFiles\\COVID-19-new.json"
    config_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\Configs\\COVID-19-config.json"

    scm = SCM(bel_file_path, config_file_path)

if __name__ == "__main__":
    main()
