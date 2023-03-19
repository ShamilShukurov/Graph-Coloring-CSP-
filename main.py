import sys
from csp import *

if len(sys.argv)==2:
  
  input_filepath = sys.argv[1]
  n_col, g_edges = read_input(input_filepath)
  c_list = generate_colors(n_col)
  g = generate_graph(g_edges)
  d = generate_domains(g, c_list)
  print("Graph is generated")
  ex = GraphColoring(edge_list = g_edges, colors = c_list, graph=g, init_domains = d )
  solution = ex.BackTracking_Search()
  if (solution == {}):
    print("No Solution")
  else:
    ex.print_solution()
  #ex.draw_solution()
else:
  print("Usage: python3 main.py <input_filepath>")
