import unittest

class TestSCM(unittest.TestCase):

    def scm_integration_test(self):

        from Neuirps_BEL2SCM.SCM import SCM

        bel_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\BELSourceFiles\\COVID-19-new.json"
        config_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\Configs\\COVID-19-config.json"

        scm = SCM(bel_file_path, config_file_path)

        bel_assertion_len = scm._json_load(bel_file_path)[0]['nanopub']['assertions']

        self.assertEqual(len(scm.graph), bel_assertion_len)
