# Graph Coloring (CSP)
Graph Coloring problem solved with Backtracking algorithm

We are given a graph in the form of a text file, that we are supposed to color.  The proper vertex coloring is such that each vertex is assigned a color and no two adjacent vertices are assigned the same color. We implemented Backtracking algorithm along with MCV, LCV, and AC-3 techniques to solve the problem. Detailed report on implementation is placed in the repo.

Input Format 

#Everything that starts with # is a comment 

#First non comment line is of form Colors = n  

Colors = 3

#Here comes the graph 

1,3 

2,18

3,19

2,19

#The “graph” presented above has 5 vertices: “1”, “2”, “3”, “18” and “19”, and 4 edges. 

#Only the edges are provided in terms of first vertex and second vertex 

#Edges are undirected: 1 is adjacent to 3, and 3 is adjacent to 1. 

#In some graphs, the edge may be included twice (1,3) as well as (3,1) – just ignore the second one.

For running the program execute following command

```python3 main.py <filepath>```

where ```filepath``` is a filepath to an input file.
