# -*- coding: utf-8 -*-

"""Constants for testing."""

from bel2scm.probability_dsl import Condition, Variable

S = Variable('S')
T = Variable('T')
Q = Variable('Q')
A = Variable('A')
A_1 = A[1]
B = Variable('B')
B_1 = B[1]
B_2 = B[2]
C = Variable('C')
D = Variable('D')
A_GIVEN = Condition(child=A, parents=[])
A_GIVEN_B = Condition(child=A, parents=[B])
A_GIVEN_B_C = Condition(child=A, parents=[B, C])
A_GIVEN_B_1 = Condition(child=A, parents=[B_1])
A_GIVEN_B_1_B_2 = Condition(child=A, parents=[B_1, B_2])
C_GIVEN_D = Condition(child=C, parents=[D])
