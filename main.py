import sys
from csp import GraphColoring

if len(sys.argv)==2:
  ex = GraphColoring(sys.argv[1])
  solution = ex.BackTracking_Search()
  if (solution == {}):
    print("No Solution")
  else:
    ex.print_solution()
  #ex.draw_solution()
else:
  print("Usage: python3 main.py <input_filepath>")
