import sys
from csp import *

if len(sys.argv)==2:
  # Reading file and creating graph
  input_filepath = sys.argv[1]
  n_col, g_edges = read_input(input_filepath)
  c_list = generate_colors(n_col)
  g = generate_graph(g_edges)
  d = generate_domains(g, c_list)
  print("Graph is generated")
  # Running backtrack algorithm
  ex = GraphColoring(edge_list = g_edges, colors = c_list, graph=g, init_domains = d )
  solution = ex.BackTracking_Search(verbose=1)
  if (solution == {}):
    print("No Solution")
  else:
    print("Solution:")
    ex.print_solution()
    ex.draw_solution()
else:
  print("Usage: python3 main.py <input_filepath>")
