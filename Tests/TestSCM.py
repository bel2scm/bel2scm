import unittest


class TestSCM(unittest.TestCase):
    # def test_something(self):
    #     self.assertEqual(True, False)

    def test_scm_integration_test(self):

        from Neuirps_BEL2SCM.SCM import SCM

        bel_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\BELSourceFiles\\COVID-19-new.json"
        config_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\Configs\\COVID-19-config.json"

        scm = SCM(bel_file_path, config_file_path)

        bel_assertions = scm._json_load(bel_file_path)[0]['nanopub']['assertions']
        bel_assertion_count = self._get_unique_name_count_from_bel_assertion(bel_assertions)

        self.assertEqual(len(scm.graph), bel_assertion_count)

    def _get_unique_name_count_from_bel_assertion(self, bel_assertions):
        names = []
        for assertion in bel_assertions:
            if assertion["subject"] not in names:
                names.append(assertion["subject"])
            if assertion["object"] not in names:
                names.append(assertion["object"])

        return len(names)

if __name__ == '__main__':
    unittest.main()
