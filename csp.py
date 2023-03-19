import networkx as nx
import matplotlib.pyplot as plt
import random

""" GENERATION FUNCTIONS
      - Used to read file and create instance of GraphColoring class
"""


"""
Function : read_input
Input: filepath of input file
Return: tuple - first element in tuple is number of colors, 
        second element is list of edges that makes graph
"""

def read_input(input_file_path: str) -> tuple:
  # Read file
  with open(input_file_path, encoding="utf8") as f:
    lines = f.readlines()
  # Filter out comment lines and empty lines
  filtered = list(filter(lambda l: l != '\n' and l[0] != '#', lines))

  # Remove newline characters from lines
  cleaned = list(map(lambda l: l.rstrip(), filtered))

  #First line shows number of colors
  n_colors = int(cleaned[0][-1])

  #Other lines shows edges
  edges = []
  for e in cleaned[1:]:
    if(e[-1]==','):
      e = e[:-1]
    edges.append(tuple(map(int, e.split(','))))

  return n_colors, edges

# Generates color list from given number of colors
def generate_colors(n: int) -> list:
  return list(range(1, n + 1))

# Generates dict for graph
def generate_graph(edges: list) -> dict:
  g = {}

  # utility function for adding j as a neighbour of i
  def add_neighbour(i: int, j: int):
    if (j is None) and (i not in g):
      g[i] = []
      return
    
    if (i in g):
      if (j not in g[i]):
        g[i].append(j)
    else:
      g[i] = [j]

  for e in edges:
    if(len(e)==2):
      add_neighbour(e[0], e[1])
      add_neighbour(e[1], e[0])
    if(len(e)==1):
      add_neighbour(e[0],None)
    
  return g

# Generate dict of domains of variables
def generate_domains(graph: dict, colors: list):
  variables = list(graph.keys())
  domain_dict = {}
  for X_i in variables:
    domain_dict[X_i] = [color for color in colors]
  return domain_dict

def copy_domains(domains: dict) -> dict:
  d_copy = {}
  for key in domains:
    d_copy[key] = [color for color in domains[key]]

  return d_copy

def shuffle_domains(domains:dict)->dict:
  for key in domains:
     random.shuffle(domains[key])
  return domains

def copy_assignments(assignments: dict) -> dict:
  a_copy = {}
  for key in assignments:
    a_copy[key] = assignments[key]

  return a_copy



class GraphColoring:

  """ BELOW METHODS ARE CLASS INSTANCE HELPER METHODS
    """

  # Constructor
  def __init__(self, edge_list, colors, graph, init_domains):
    # n_col, g_edges = read_input(input_file_path)
    # c_list = generate_colors(n_col)
    # g = generate_graph(g_edges)
    # d = generate_domains(g, c_list)

    self.edge_list = edge_list
    self.colors = colors
    self.graph = graph
    self.variables = list(graph.keys())
    self.init_domains = init_domains
    self.solution = {}

  # For printing object info
  def __str__(self):
    return "Variables: {}\nColors: {}\nGraph:{}".format(
      self.variables, self.colors, self.graph)

  # Get neighbours of specified variable
  def neighbours(self, X_i: int) -> list:
    if X_i in self.graph:
      return self.graph[X_i]

  # Remove specified value from the domain of specified variable
  def reduce_domain(self, domains: dict, val: int, X_i: int):
    d = copy_domains(domains)
    if X_i in d:
      if val in d[X_i]:
        d[X_i].remove(val)
    return d

  # Constraint : check if X_i and X_j has same color
  def _constraint(self, assignments: dict, X_i: int, X_j: int):
    j_col = -1
    if X_j in assignments:
      j_col = assignments[X_j]
    return j_col != assignments[X_i]

  # Calculate partial weight: Given node's color is different from all its neighbours'
  def _partial_weight(self, assignments: dict, X_i: int):
    for neighbour in self.neighbours(X_i):
      c = self._constraint(assignments, X_i, neighbour)
      if (not c):
        return 0
    return 1

  # Check if assignments are Complete and Correct
  def check_complete(self, assignments: dict):
    for X_i in self.variables:
      # Check if all vertices are colored
      if X_i not in assignments:
        return 0

      # Check if all partial_weights are 1 (no adjacent vertices has same color)
      if self._partial_weight(assignments, X_i) == 0:
        return 0
    return 1

  """ BELOW METHODS IS BUILT FOR IMPLEMENTING BACKTRACKING AND ITS PARTS : MCV,LCV,AC_3
    """

  # Select most constrained variable
  # In case of tie take the one that has most neighbours
  def MCV(self, assignments: dict, domains: dict):
    # select all unassigned variables
    left_variables = list(
      filter(lambda x: x not in assignments, self.variables))

    # generate dictionary that holds neighbour count of each variable
    lv = {}
    for var in left_variables:
      lv[var] = len(self.graph[var])

    # sort lv by neighbour count
    sorted_lv = dict(sorted(lv.items(), key=lambda x: x[1], reverse=True))

    left_variables = list(sorted_lv.keys())

    #find minimum constrained variable
    min_var = left_variables[0]
    min_c = len(domains[min_var])

    for X_i in left_variables[1:]:
      if (len(domains[X_i]) < min_c):
        min_c = len(domains[X_i])
        min_var = X_i
    return min_var

  def LCV(self, assignments: dict, domains: dict, X_i: int):
    colors_domain = []
    color_index = 0
    for col in domains[X_i]:
      cv = 0  #consistent values
      for X_j in self.neighbours(X_i):
        cv = cv + len(domains[X_j])
        if (col in domains[X_j]):
          cv = cv - 1
      colors_domain.append((cv, color_index, col))
      color_index = color_index+1

    colors_domain.sort(reverse=True)

    ordered_domain_values = list(map(lambda x: x[2], colors_domain))

    return ordered_domain_values

  #Remove values from domain of X_i to make X_i arc consistent with respect to X_j.
  # In our specific graph coloring problem, enforce only happens if there is only one value in the domain of X_j
  def enforce_arc_consistency(self, domains: dict, X_i: int, X_j: int):
    d_copy = copy_domains(domains)

    if (len(d_copy[X_j]) == 0):
      return None
    if (len(d_copy[X_j]) == 1):
      val = d_copy[X_j][0]
      self.reduce_domain(d_copy, val, X_i)
    return d_copy

  # we will call AC_3 after coloring X_i
  def AC_3(self, domains: dict, X_j: int):

    domains_old = copy_domains(domains)
    domains_new = copy_domains(domains)

    #print("domains_old: {}\ndomains_new: {}".format(domains_old,domains_new))

    S = [X_j]

    while (len(S) > 0):
      X_j = S.pop(0)

      for X_i in self.neighbours(X_j):
        domains_new = self.enforce_arc_consistency(domains_old, X_i, X_j)
        if (domains_new == None):
          return None
        #print("Domains Old: {}\nDomains_New: {}".format(domains_old,domains_new))
        if (len(domains_old[X_i]) != len(domains_new[X_i])
            and len(domains_new[X_i]) == 1):
          S.append(X_i)
        domains_old = domains_new.copy()
    return domains_new

  def BackTracking(self, assignments, domains):
    #print("Assignments: ",assignments)
    #print("Domains: ",domains)

    if self.check_complete(assignments) == 1:
      return assignments

    assignments_new = copy_assignments(assignments)

    X_i = self.MCV(assignments_new, domains)

    ordered_vals = self.LCV(assignments_new, domains, X_i)

    for v in ordered_vals:

      assignments_new[X_i] = v

      pw = self._partial_weight(assignments_new, X_i)

      if (pw == 0):
        #                 print("Constraints do not meet for variable {} and value {}".format(X_i,v))
        assignments_new = copy_assignments(assignments)
        continue

      domains_new = copy_domains(domains)
      domains_new[X_i] = [v]

      domains_new = self.AC_3(domains_new, X_i)

      #If any domains_i is empty, continue
      if domains_new is None or any(
          len(domains_new[d_i]) == 0 for d_i in domains_new):
        #                 print("AC_3 do not meet for variable {} and value {}".format(X_i,v))
        assignments_new = copy_assignments(assignments)
        domains_new = copy_domains(domains)
        continue

      result = self.BackTracking(assignments_new, domains_new)
      #print("after recursive call, result:", result)
      if result != {}:
        return result

    return {}

  def BackTracking_Search(self):
    assignments = {}
    domains = copy_domains(self.init_domains)
    domains_shuffles = shuffle_domains(domains)
    res = self.BackTracking(assignments, domains_shuffles)
    self.solution = res
    return res

  """METHODS FOR DISPLAYING THE RESULTS"""

  def print_solution(self):
    if self.solution == {}:
      print("No Solution")
      return
    for node in self.solution:
      print("Node {} colored with {}".format(node, self.solution[node]))

  def draw_solution(self):

    not_isolates = list(filter(lambda edge: len(edge)==2, self.edge_list))
    isolates = list(filter(lambda edge: len(edge)==1, self.edge_list))
    G = nx.from_edgelist(not_isolates)
    for i in isolates:
      G.add_node(i[0])

    

    if self.solution == {}:
      colors = "blue"
      colormap = None
      title = "No Solution!"
      
    else:
      colors = [self.solution[n] for n in G.nodes()]
      colormap = plt.cm.Set1
      title = "Solution exists:"
      

    plt.figure(figsize=(6, 4))
    pos = nx.kamada_kawai_layout(G)

    node_options = {"node_color": colors, "node_size": 100}
    edge_options = {"width": 0.5, "alpha": 1, "edge_color": "black"}

    node_label_options = {
      "font_size": 20,
      "font_color": "black",
      "verticalalignment": "bottom",
      "horizontalalignment": "left"
    }
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, **node_options, cmap=colormap)

    # Draw edges
    nx.draw_networkx_edges(G, pos, **edge_options)

    # Draw your labels
    nx.draw_networkx_labels(G, pos, **node_label_options)

    plt.axis("off")
    plt.title(title)
    
    plt.show()
