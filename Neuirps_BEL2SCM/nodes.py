class Node():
    LABEL_DICT = {
        'transformation': ['sec', 'surf', 'deg', 'rxn', 'tloc', 'fromLoc',
                           'products', 'reactants', 'toLoc'],
        'abundance': ['a', 'abundance', 'complex', 'complexAbundance', 'geneAbundance', 'g',
                      'microRNAAbundance', 'm', 'populationAbundance', 'pop', 'proteinAbundance', 'p',
                      'rnaAbundance', 'r', 'frag', 'fus', 'loc', 'pmod', 'var'
                                                                         'compositeAbundance', 'composite'],
        'activity': ['activity', 'act', 'molecularActivity', 'ma'],
        'reaction': ['reaction', 'rxn'],
        'process': ['biologicalProcess', 'bp'],
        'pathology': ['pathology', 'path']
    }

    def __init__(self):
        self.root = True
        self.name = ""
        self.children = []
        self.parent_relations = []
        self.child_relations = []
        self.node_type = ""
        self.node_label = ""
        self.children_type = []
        self.children_label = []
        self.parents = []
        self.parent_type = []
        self.parent_label = []

    def get_node_information(self, sub, obj, rel):
        if rel.find('crease') > 0:
            self.name = sub
            self.children.append(obj)
            self.child_relations.append(rel)

            p = sub
            c = obj
            ptype = p[:p.find('(')]
            ctype = c[:c.find('(')]

            self.node_type = ptype
            self.children_type.append(ctype)

            for label in self.LABEL_DICT:
                if ptype in self.LABEL_DICT[label]:
                    self.node_label = label
                elif ptype == '':
                    self.node_label = 'Others'
            for label in self.LABEL_DICT:
                if ctype in self.LABEL_DICT[label]:
                    self.children_label.append(label)
                elif ctype == '':
                    self.children_label.append('Others')
                    break

    def update_parent_node(self, obj, rel):
        if rel.find('crease') > 0:
            self.children.append(obj)
            self.child_relations.append(rel)

            c = obj
            ctype = c[:c.find('(')]
            self.children_type.append(ctype)
            for label in self.LABEL_DICT:
                if ctype in self.LABEL_DICT[label]:
                    self.children_label.append(label)
                elif ctype == '':
                    self.children_label.append('Others')
                    break

    def update_child_node(self, sub, rel):
        self.root = False
        if rel.find('crease') > 0:
            self.parents.append(sub)
            self.parent_relations.append(rel)
            p = sub
            ptype = p[:p.find('(')]
            self.parent_type.append(ptype)
            for label in self.LABEL_DICT:
                if ptype in self.LABEL_DICT[label]:
                    self.parent_label.append(label)
                elif ptype == '':
                    self.parent_label.append('Others')
                    break

    def __eq__(self, other):
        if self.root == other.root and \
                self.name == other.name and \
                self.children == other.children and \
                self.parent_relations == other.parent_relations and \
                self.child_relations == other.child_relations and \
                self.node_type == other.node_type and \
                self.node_label == other.node_label and \
                self.children_type == other.children_type and \
                self.children_label == other.children_label and \
                self.parents == other.parents and \
                self.parent_type == other.parent_type and \
                self.parent_label == other.parent_label:
            return True
        else:
            return False
