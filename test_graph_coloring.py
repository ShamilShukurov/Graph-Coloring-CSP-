import unittest
from csp import *

class TestGraphColoring(unittest.TestCase):
  
  def setUp(self):
    # Add setup code here
    edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    colors = [1,2,3]
    graph = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
    init_domains = {0: [1,2,3], 1: [1,2,3], 
                    2: [1,2,3], 3: [1,2,3]}
    self.gc = GraphColoring(edge_list, colors, graph, init_domains)

  def test_AC_3(self):
    # We coloured 0 with 1, now we will call AC_3 on 0
    domains = {0: [1], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]}
    domains_new = self.gc.AC_3(domains, 0)
    
    # Assert the output is as expected
    expected_domains = {0: [1], 1: [2, 3], 2: [2, 3], 3: [1,2,3]}
    self.assertEqual(domains_new, expected_domains)

  def test_check_complete(self):
    #Test if check_complete returns 0 if the assignments are incomplete or not correct, 
    #and 1 otherwise
    assignments = {0: 1, 1: 2, 2: 3, 3: 1}
    self.assertEqual(self.gc.check_complete(assignments), 1)
    assignments = {0: 1, 1: 2, 2: 3, 3: 2}
    self.assertEqual(self.gc.check_complete(assignments), 0)

  def test_reduce_domain(self):
    #Test if reduce_domain remove the specified value from the domain of the specified variable
    domains = {0: [1,2,3], 1: [1,2,3], 2: [1,2,3], 3: [1,2,3]}
    new_domains = self.gc.reduce_domain(domains, 1, 1)
    self.assertEqual(new_domains[1], [2,3])

  def test_LCV(self):
    # Test if LCV returns the domain of the variable ordered 
    # by decreasing number of consistent values of neighboring variables
    # if tie LCV returns reverse order of values
    assignments = {3:1}
    domains = {0: [1,2,3], 1: [2, 3], 2: [2, 3], 3: [1]}
    self.assertEqual(self.gc.LCV(assignments, domains, 0), [1,3,2])
  
  def test_MCV(self):
    assignments = {0:1}
    domains = {0:1, 1:[2,3], 2:[2,3], 3:[1,2,3]}
    self.assertEqual(self.gc.MCV(assignments, domains), 1)
  
  
  def test_setup_graph(self):
    # perform graph coloring with backtracking and MCV, LCV, AC_3
    solution = self.gc.BackTracking_Search(verbose=0)
  
    # assert that solution is valid
    self.assertTrue(self.gc.check_complete(solution))
  
  
  def test_impossible_graph(self):
      # initialize graph
      edge_list = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5),(4,5),(5,6),(5,7),(5,8),(6,7),(6,8),(7,8)]
      colors = [1,2,3]
      graph = {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4, 5], 4: [1, 2, 3, 5], 5: [3, 4, 6, 7, 8], 6: [5, 7, 8], 7: [5, 6, 8], 8: [5, 6, 7]}
      init_domains = {1: [1,2,3], 2: [1,2,3], 3: [1,2,3], 4: [1,2,3], 5: [1,2,3], 6: [1,2,3], 7: [1,2,3], 8: [1,2,3]}
  
      # initialize graph coloring object
      g = GraphColoring(edge_list, colors, graph, init_domains)
  
      # perform graph coloring with backtracking and MCV, LCV, AC_3
      solution = g.BackTracking_Search(verbose=0)
  
      # assert that solution is valid
      self.assertFalse(g.check_complete(solution))
  
  
  def test_simple_graph(self):
    # Enter code here
      # initialize graph
      edge_list = [(1,2),(2,3),(3,4),(4,1)]
      colors = [1,2,3]
      graph = {1: [2, 4], 2: [1, 3], 3: [2, 4], 4: [1, 3]}
      init_domains = {1: [1,2,3], 2: [1,2,3], 3: [1,2,3], 4: [1,2,3]}
  
      # initialize graph coloring object
      g = GraphColoring(edge_list, colors, graph, init_domains)
  
      # perform graph coloring with backtracking and MCV, LCV, AC_3
      solution = g.BackTracking_Search(verbose=0)
  
      # assert that solution is valid
      self.assertTrue(g.check_complete(solution))


if __name__ == '__main__':
    unittest.main()




