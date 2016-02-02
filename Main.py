import Download
import codecs
from ACO.Ant import ANT
from ACO.Node import Node
from ACO.ACO import ACO
from ACO.Edge import Edge
from ACO import CityFilter
import random
import numpy as np
import ACO.CityFilter
import ACO.CUDA
import time

world_cities_path = "./Download/worldcitiespop.txt.gz"
world_cities_txt_path = "./Download/worldcitiespop.txt"

# Download world_cities
#world_cities_path = Download.download("http://download.maxmind.com/download/worldcities/worldcitiespop.txt.gz", "./Download/")

# Decompress
#world_cities_txt_path = Download.gz(world_cities_path)

# Download country_codes
country_codes_path = Download.download("http://data.okfn.org/data/core/country-list/r/data.csv", "./Download/")


def load_country_codes():
    country_codes = {}
    with open(country_codes_path) as file:
        for row in file.readlines():
            row = row.replace("\n", "").replace("\"","")
            split = row.rsplit(',', 1)
            #country_codes[row[1]] = row[0]
    return country_codes

def load_world_cities(loc=["*"], max=None):
    nodes = []
    nodes_latitude = []
    nodes_longitude = []

    with codecs.open(world_cities_txt_path, "r",encoding='utf-8', errors='ignore') as file:
        items = file.readlines()[1:]

        i=0
        for row in items:
            # Country, City, AccentCity, Region, Population, Latitude, Longitude
            split = row.split(",")
            c_code = str(split[0])
            city = str(split[1])
            lat = str(split[5])
            lon = str(split[6].replace("\n",""))

            if "*" in loc or c_code in loc:
                nodes.append(Node(idx=i, country=c_code, city=city, lat=lat, lon=lon))
                nodes_latitude.append(lat)
                nodes_longitude.append(lon)
                i += 1

                if max != None and max <= i:
                    break

    return nodes, np.array(nodes_latitude, dtype=np.float32), np.array(nodes_longitude, dtype=np.float32)

# Load country codes
start = time.time()
country_codes = load_country_codes()
print("[+{0}] Loaded country_code converter!".format(round(time.time() - start, 2)))

# Load Nodes
start = time.time()
nodes, nodes_latitude, nodes_longitude = load_world_cities([ "no"], max=512)
print("[+{0}] Parsed {1} nodes.".format(round(time.time() - start, 2), len(nodes)))

# Create edges
start = time.time()
edges = ACO.CUDA.create_distance_matrix(nodes_latitude, nodes_longitude)
print("[+{0}] Generated {1} edges using GPU".format(round(time.time() - start, 2), len(edges) ** 2))

MAX_PHEROMONES = 1000
MIN_PHEROMONES = 1

# Create edge_pheromones
start = time.time()
edges_pheromones = np.ones((nodes_latitude.shape[0], nodes_longitude.shape[0]), dtype=np.float32)
edges_pheromones.fill(MAX_PHEROMONES)
np.fill_diagonal(edges_pheromones, 0)
print("[+{0}] Created pheromones map".format(round(time.time() - start, 2)))

MAX_COST = np.sum(edges) # TODO , some nan values. BUT WHERE?


start_node = nodes[0]
target_node = nodes[50]


edges[start_node.idx][target_node.idx] = 100000 # 599939393


result = {}
cost_result_x = []
cost_result_y = []

lowest_cost_path = ""
lowest_cost_val = 10000000000000000000000000000

for i in range(10000):

    ant = ANT(start_node, nodes, edges, edges_pheromones, MAX_COST=MAX_COST, MAX_PHEROMONES=MAX_PHEROMONES, MIN_PHEROMONES=MIN_PHEROMONES, MAX_STEPS=10, target_node=target_node)
    goal = ant.walk()
    ant.pheromones()

    # MMAS Decay
    edges_pheromones = np.multiply(edges_pheromones, .99)




    cost_sum = sum([float(edges[item[0]][item[1]]) for item in ant.visited_edges])
    path_length = len(ant.visited_edges)


    cost_result_y.append(cost_sum)
    cost_result_x.append(i)

    #print("[{2}]: Cost: {0} | P_Length: {1}".format(cost_sum, path_length, i))
    #print("{0} | {1} | {2}".format(ant.visited_edges, edges_pheromones[0][7], cost_sum))

    identifier = str(ant.visited_edges[0][0])
    for visited in ant.visited_edges:
        identifier += " => " + str(visited[1])


    if cost_sum < lowest_cost_val:
        lowest_cost_val = cost_sum
        lowest_cost_path = identifier


    try:
        result[identifier]["num"] += 1
    except:
        result[identifier] = {
            'num': 1,
            'cost': cost_sum,
            'path_length': path_length

        }



import numpy as np
import matplotlib.pyplot as plt


for key in sorted(result, key=lambda x: (result[x]['num'], result[x]['cost'])):
    val = result[key]
    print(key)
    print("\tNum: " + str(val["num"]))
    print("\tCost: " + str(val["cost"]))
    print("\tPath_Length: " + str(val["path_length"]))

print("--------------------")
print("Path: " + str(lowest_cost_path))
print("Cost: " + str(lowest_cost_val))


plt.plot(cost_result_x, cost_result_y, 'ro')
plt.show()





