# -*- coding: utf-8 -*-

"""Constants for testing."""

from bel2scm.probability_dsl import ConditionalProbability, Variable, Intervention, CounterfactualVariable

S = Variable('S')
T = Variable('T')
Q = Variable('Q')
A = Variable('A')
#A_1 = A[1]
B = Variable('B')
#B_1 = B[1]
#B_2 = B[2]
C = Variable('C')
D = Variable('D')
A_GIVEN = ConditionalProbability(child=A, parents=[])
A_GIVEN_B = ConditionalProbability(child=A, parents=[B])
A_GIVEN_B_C = ConditionalProbability(child=A, parents=[B, C])
#A_GIVEN_B_1 = ConditionalProbability(child=A, parents=[B_1])
#A_GIVEN_B_1_B_2 = ConditionalProbability(child=A, parents=[B_1, B_2])
C_GIVEN_D = ConditionalProbability(child=C, parents=[D])

Y = Variable('Y')
W = Variable('W')
X = Variable('X')
Z = Variable('Z')
