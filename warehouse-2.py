import math
import re
from collections import defaultdict  
from heapq import * 
import time
import numpy
from tkinter import *

WareHouse = open("warehouse-grid.csv", "r")
ItemInfo = open("item-dimensions-tabbed.txt")
maxWeight = math.inf
item_info = {}
items_list = {}
id_list = {}
maxX, maxY = 0, 0
gridw, gridh = 0, 0
edges = []
shelf = []
RUNNING_TIME = 1000 # allowed running time for algorithm (ms), for example, 30000ms = 30 seconds

################################################################################################## Methods ###############################################################################################

# dijksta algorithm for shortest path
# inspired by posts on https://stackoverflow.com/questions/22897209/dijkstras-algorithm-in-python
def dijkstra(edges, start, end):  
    graph = defaultdict(list)  
    for n1,n2,distance in edges:  
        graph[n1].append((distance,n2))  
    Q = [(0, start, ())]
    visited = set()  
    while Q:  
        (dis, node1, path) = heappop(Q)  
        if node1 not in visited:  
            visited.add(node1)  
            path = (node1, path)  
            if node1 == end:  
                return dis, path  
            for distance, node2 in graph.get(node1, ()):  
                if node2 not in visited:  
                    heappush(Q, (dis + distance, node2, path))  
    return math.inf, []

def getPath(edges, start, end):  
	if start == end:
		return 0, [start, end]
	else:
	    path_len = math.inf  
	    path = []  
	    length, path_dijk = dijkstra(edges, start, end)  
	    if path_dijk: 
	    	path_len = length
	    	while path_dijk:
	        	node = path_dijk[0]
	        	path.append(node)
	        	path_dijk = path_dijk[1]
	    	path.reverse()
	    	return path_len, path

def shortestPath(edges, start, itemLocation):
	if (start - 1 == itemLocation) or (start + 1 == itemLocation):
		return 0, [start, start], start
	else: 
		left = itemLocation - 1;
		right = itemLocation + 1;
		length1, path1 = getPath(edges, start, left)
		length2, path2 = getPath(edges, start, right)
		if(length1 <= length2):
			return length1, path1, left
		else:
			return length2, path2, right

def convert(x, y):
	return x + 1 + y * (maxX + 2)

def mass_convert(positions):
	result = []
	for each in positions:
		result.append(convert(each))
	return result

def rev_convert(position):
	position -= 1
	return [position % gridw, position // gridw]

def mass_rev_convert(positions):
	result = []
	for i in range(len(positions)):
		result.append(rev_convert(positions[i]))
	return result

def order_convert(orderList):
	result = []
	for i in range(len(orderList)):
		result.append(items_list[orderList[i]])
	return result

def order_rev(orderList):
	result = []
	for i in range(len(orderList)):
		result.append(id_list[orderList[i]])
	return result

def default_Order(orderlist, edges, start, end):
	totalLength = 0
	totalPath = [start]
	fetchpoints = []
	pt = -1
	for i in range(len(orderlist)):
		nextEnd = orderlist[i]
		length, path, pt = shortestPath(edges, start, nextEnd)
		totalLength += length
		totalPath.extend(path[1:])
		fetchpoints.append(pt)
		start = pt
	length, path = getPath(edges, pt, end)
	totalLength += length;
	totalPath.extend(path[1:])
	return totalLength, totalPath, fetchpoints

def calculate(pt1, pt2):
	pt1 = rev_convert(pt1)
	pt2 = rev_convert(pt2)
	return (pt1[0] - pt2[0])^2 + (pt1[1] - pt2[1])^2

def lazyNearest(orders, orderids, start):
	minLength = math.inf
	nearOrder = None
	nearestid = None
	temp = None
	for i in range(len(orders)):
		position = orders[i]
		dist = calculate(start, position)
		if(dist < minLength):
			minLength = dist
			nearOrder = position
			nearestid = orderids[i]
	return nearOrder, nearestid

def opti_Order(orderlist, orderids, edges, start, end): # Nearest neighbor algorithm
	totalLength = math.inf
	totalPath = []
	fetchpoints = []
	newOrder = []
	st_time = time.time()
	for i in range(0, len(orderlist)):
		if (time.time() - st_time) * 1000 >= RUNNING_TIME: # limiting running time
			break
		oneLength, onePath, st = shortestPath(edges, start, orderlist[i]) #distance info to a certain item
		oneFetchpts = [st]
		oneOrder = [orderids[i]]
		restlist = list(orderlist)
		restlist.remove(orderlist[i])
		restids = list(orderids)
		restids.remove(orderids[i])
		while(restlist):
			nearest, nearestid = lazyNearest(restlist, restids, st)
			length, path, pt = shortestPath(edges, st, nearest)
			oneLength += length
			onePath.extend(path[1:])
			oneFetchpts.append(pt)
			st = pt
			restlist.remove(nearest)
			restids.remove(nearestid)
			oneOrder.append(nearestid)
		length, path = getPath(edges, st, end)
		oneLength += length;
		onePath.extend(path[1:])
		
		if totalLength > oneLength:
			totalLength = oneLength
			fetchpoints = oneFetchpts
			totalPath = onePath
			newOrder = oneOrder	

	return totalLength, totalPath, fetchpoints, newOrder


def getDir(pt1, pt2):
	direction = None
	row = (pt1 - 1) // gridw
	lower = row * gridw + 1
	upper = (row + 1) * gridw
	if pt1 == pt2:
		direction = 5
	elif (pt2 <= upper) and (pt2 >= lower): #on the same row
		if pt2 > pt1:
			direction = 6
		else:
			direction = 4
	elif((pt1 - 1) % gridw) == ((pt2 - 1) % gridw): #on the same column
		if pt2 > pt1:
			direction = 2
		else:
			direction = 8
	return direction

def pathPrint(path, pickPts):
	i = 1 
	while(i < len(path) - 1):
		left = getDir(path[i-1], path[i])
		right = getDir(path[i], path[i+1])
		if(left == right) and (not path[i] in pickPts) and (left != None) and (right != None):
			path.pop(i)
		else:
			i += 1
	return path	

temp = []

def filePrint(file, path, pickPtsSet, items):
	count = 0
	pickPts = []
	pickPts.extend(pickPtsSet)
	path = pathPrint(path, pickPts)
	i = 0
	newLine = True
	goto = False
	while i < len(path):
		current = rev_convert(path[i])
		if(newLine):
			file.write("From " + str(current))
			i += 1
			newLine = False
			goto = True
		elif(path[i] in pickPts):
			if(goto):
				file.write(", go to " + str(current) + ". pick up " + str(items[count]) + " from " + str(current) + ".\n")
				goto = False
			else: 
				file.write(", then go to " + str(current) + ". pick up " + str(items[count]) + " from " + str(current) + ".\n")
			count += 1;
			pickPts.remove(path[i])
			newLine = True
		else:
			if(goto):
				file.write(", go to " + str(current))
				goto = False
			else:
				file.write(", then go to " + str(current))
			i += 1
		if ((not newLine) and i == len(path)): file.write(".\n")

def sysPrint(path, pickPtsSet, items):
	count = 0
	pickPts = []
	pickPts.extend(pickPtsSet)
	path = pathPrint(path, pickPts)
	i = 0
	newLine = True
	goto = False
	while i < len(path):
		current = rev_convert(path[i])
		if(newLine):
			print("From " + str(current), end="")
			i += 1
			newLine = False
			goto = True
		elif(path[i] in pickPts):
			if(goto):
				print(", go to " + str(current) + ". Pick up " + str(items[count]) + " from " + str(current) + ".")
				goto = False
			else: 
				print(", then go to " + str(current) + ". Pick up " + str(items[count]) + " from " + str(current) + ".")
			count += 1;
			pickPts.remove(path[i])
			newLine = True
		else:
			if(goto):
				print(", go to " + str(current), end="")
				goto = False
			else:
				print(", then go to " + str(current), end="")
			i += 1
		if ((not newLine) and i == len(path)): print(".")

def reduceMatrix(matrix):
	m = matrix
	cost = 0
	height = m.shape[0]
	width = m.shape[1]

	for i in range(height):
		minimum = min(m[i])
		if(minimum != math.inf):
			cost += minimum
			m[i] -= minimum

	for i in range(width):
		minimum = min(m[:,i])
		if(minimum != math.inf):
			cost += minimum
			m[:,i] -= minimum

	return m, cost

def nextStep(matrix, source, destination, sdcost):
	m  = matrix
	m[source] = math.inf
	m[:,destination] = math.inf
	m[source][destination] = math.inf
	m[destination][source] = math.inf
	nextm, nextcost =  reduceMatrix(m)
	return nextm, nextcost + sdcost

class matrixNode:
	def __init__(self, oid, matrix, root, children, cost):
		self.oid = oid
		self.matrix = matrix
		self.root = root
		self.children = children
		self.cost = cost

def otherid(each):
	if each == 0:
		return 0;
	elif each % 2 == 0:
		return each - 1
	else:
		return each + 1 

def bnb(pathLens, orderSet, start, orderList):
	current = start
	conti = True
	searchMin = [current]
	tmp = start
	nextorders = []
	globalMin = None
	st_of_loop = time.time()

	while(conti and (time.time() - st_of_loop) * 1000 < RUNNING_TIME):

		#find min cost node
		currentMin = searchMin[0]
		for each in searchMin:
			if each.cost < currentMin.cost:
				currentMin = each
			elif each.cost == currentMin.cost:
				if len(each.root) > len(currentMin.root):
					currentMin = each 	
		current = currentMin

		path_to_current = current.root
		path_to_current.append(current.oid)

		#get left acesses for the rights, right accesses for the lefts
		othersides = []
		for each in path_to_current:
			othersides.append(otherid(each))

		#list all unvisited
		nextorders = orderSet - set(current.root) - set([current.oid]) - set(othersides)

		if len(nextorders) != 0:
			for each in nextorders:
				m, cost = nextStep(current.matrix, current.oid, each, pathLens[current.oid][each])
				tmp = matrixNode(each, m, path_to_current, [], cost)
				current.children.append(tmp)
				searchMin.append(tmp)
			searchMin.remove(current)
		else:
			# reach one solution 
			# loop till timeout
			if globalMin == None:
				globalMin = current
			if globalMin.cost > current.cost:
				globalMin = current
			if len(searchMin) == 0: 
				conti = False

	tmp = list(globalMin.root)
	tmp.append(globalMin.oid)

	newOrder = []
	for i in range(1, len(tmp)):
		a = int(tmp[i] / 2)
		a = tmp[i] % 2 + a
		newOrder.append(orderList[a - 1])
	newOrder2 = []
	for each in newOrder:
		if each not in newOrder2:
			newOrder2.append(each)	
	return newOrder2

def expand(positions, start):
	tmp = [start]
	for each in positions:
		tmp.extend([each - 1, each + 1])
	return tmp

def matrixGen(positions, edges, start):
	positions = expand(positions, start)
	w = len(positions)
	m = numpy.zeros((w, w))
	m += math.inf

	for y in range(1, w):
		for x in range(0, y):
			l = abs(calculate(positions[y], positions[x]))
			#l, p = getPath(edges, positions[y], positions[x])
			m[y][x] = l
			m[x][y] = l

	return m, positions

def genPlot(start, end, route, fetch, items):
	BOXSIZE = 20
	w = BOXSIZE * gridw
	h = BOXSIZE * gridh
	master = Tk()
	w = Canvas(master, width=w, height=h)
	w.pack()

	count = 1
	for y in range(gridh):
		for x in range(gridw):
			stx = BOXSIZE * x
			sty = BOXSIZE * y
			endx = BOXSIZE * (x + 1)
			endy = BOXSIZE * (y + 1)
			if count == start:
				w.create_rectangle(stx, sty, endx, endy, fill="green")
			elif count == end:
				w.create_rectangle(stx, sty, endx, endy, fill="red")
			elif count in fetch:
				w.create_rectangle(stx, sty, endx, endy, fill="blue")
			elif count in items:
				w.create_rectangle(stx, sty, endx, endy, fill="cyan")
			elif count in shelf:
				w.create_rectangle(stx, sty, endx, endy, fill="yellow")
			elif count in route:
				w.create_rectangle(stx, sty, endx, endy, fill="purple")
			else:
				w.create_rectangle(stx, sty, endx, endy, fill="gray")
			count += 1

	mainloop()

def batchWrite(start, end, route, fetch, items, file, count, effort, newOrder):
	file.write("#" + str(count) + "\n")
	file.write(str(start) + "\n")
	file.write(str(end) + "\n")
	file.write(str(effort) + "\n")
	for each in route:
		file.write(str(each) + "\t")
	file.write("\n")
	for each in fetch:
		file.write(str(each) + "\t")
	file.write("\n")
	for each in items:
		file.write(str(each) + "\t")
	file.write("\n")
	for each in newOrder:
		file.write(str(each) + "\t")
	file.write("\n")

def batchRead(file, count):
	line = file.readline()
	while(line != "#" + str(count) + "\n"):
		line = file.readline()

	start = int(file.readline())
	end = int(file.readline())
	effort = float(file.readline())
	route = file.readline().split("\t")
	del route[-1]
	route = list(map(int, route))
	fetch = file.readline().split("\t")
	del fetch[-1]
	fetch = list(map(int, fetch))
	items = file.readline().split("\t")
	del items[-1]
	items = list(map(int, items))		
	newOrder = file.readline().split("\t")
	del newOrder[-1]
	newOrder = list(map(int, newOrder))
	return start, end, route, fetch, items, effort, newOrder

def orderTotalWeight(order):
	result = 0;
	for each in order:
		if each in item_info:
			tmp = item_info[each]
			result += tmp[3]
	return result

def getWeight(item):
	result = 0
	if item in item_info:
		tmp = item_info[item]
		result = tmp[3]
	return result

def getMinWeight(order):
	minWei = math.inf
	minItem = None
	for each in order:
		wei = getWeight(each)
		if  wei < minWei:
			minWei = wei
			minItem = each
	return minItem, minWei

def getTotalWeight(order):
	result = 0
	for each in order:
		result += getWeight(each)
	return result	

def splitOrder(order):
	result = []
	tmp = list(order)
	subOrder = []
	currentWei = 0

	while tmp:
		mini, wei = getMinWeight(tmp)
		tmp.remove(mini)
		if currentWei + wei <= maxWeight:
			currentWei += wei
			subOrder.append(mini)
		else:
			result.extend([subOrder])
			subOrder = [mini]
			currentWei = wei

	if subOrder:
		result.extend([subOrder])
	return result

def splitNmerge(order_list):
	orders = list(order_list)
	result = []
	for eachOrder in orders:
		orderWei = getTotalWeight(eachOrder)
		orders.remove(eachOrder)
		if orderWei > maxWeight:
			#split
			for e in splitOrder(eachOrder):
				result.append(e)
		elif orderWei < maxWeight:
			#merge with others
			for e in orders:
				wei = getTotalWeight(e)
				if wei + orderWei <= maxWeight:
					orderWei = wei + orderWei
					eachOrder.extend(e)
					orders.remove(e)
			result.append(eachOrder)
		else:
			#no change
			result.append(eachOrder)
	return result		
############################################################################################# Main #######################################################################################################

# read file
for line in WareHouse:
	number,x,y = line.split(", ")
	number = int(number)
	x = math.floor(float(x)) * 2 + 1
	y = int(y) * 2 + 1
	if maxX < x:
		maxX = x
	if maxY < y:
		maxY = y
	temp.append([x, y, number])
WareHouse.close()

ItemInfo.readline()
for line in ItemInfo:
	number, l, w, h, wei = line.split("\t")
	number = int(number)
	l = float(l)
	w = float(w)
	h = float(h)
	wei = float(wei)
	item_info[number] = [l, w, h, wei]
ItemInfo.close() 

gridw, gridh = maxX + 2, maxY + 2

for each in temp:
	idn = convert(each[0], each[1])
	shelf.append(idn)
	items_list[each[2]] = idn
	id_list[idn] = each[2]

# initalize edges
for i in range(gridh):
	for j in range(gridw):
		idn = convert(j, i)
		if (idn not in shelf):
			down = idn + gridw
			top = idn - gridw
			left = idn - 1
			right = idn + 1

			if (top >= 1) and (top not in shelf):
				edges.append([idn, top, 1])
			if (down <= (gridh) * (gridw)) and (down not in shelf):
				edges.append([idn, down, 1])
			if ((idn % (gridw))) != 1 and (left not in shelf):
				edges.append([idn, left, 1])
			if ((idn % (gridw)) != 0) and (right not in shelf):
				edges.append([idn, right, 1])

print("Hello User, do you wish to read history data? (1)yes (2)no")
choice0 = input()
choice0 = choice0 == "1"
if choice0:
	print("What is the order number?")
	orderN = int(input())
	print("Generating plot...")
	file = open("BatchFile.txt", "r")
	start, end, route, fetch, items, effort, newOrder = batchRead(file, orderN)
	path = [start]
	path.extend(route)
	path.append(end)
	sysPrint(path, list(fetch), newOrder)
	print("Here is the total effort:", effort)
	genPlot(start, end, route, fetch, items)
else:	
	print("Hello User, how much weight can you push?")	
	maxWeight = float(input())
	print("Hello User, where is your worker? (x, y)")
	st = input()
	st = st[1 : len(st) - 1].split(", ")
	stx = int(st[0])
	sty = int(st[1])
	print("What is Your Worker's End Location? (x, y)")
	de = input()
	de = de[1 : len(de) - 1].split(", ")
	dex = int(de[0])
	dey = int(de[1])

	start = convert(stx, sty)
	end = convert(dex, dey)

	print("Do you wish to (1) enter an order manually or (2) specify an input file? [enter 1 or 2]")
	choice = input()
	choice = choice == "1"
	print("Do you wish to use (1) nearest neightbor algorithm or (2) branch and bound? [enter 1 or 2]")
	choice2 = input()
	choice2 = choice2 == "1"

	if(choice):
		orderList = []
		splited = False
		print("Hello User, what items would you like to pick?")
		get = input().split(", ")
		for each in get:
			orderList.append(int(each))
		splitedOrders = splitOrder(orderList)
		if len(splitedOrders) > 1:
			print("This order has been splited due to weight limit!")
			splited = True
		print("Computing...")
		st_time = time.time() * 1000.0
		for eachOrder in splitedOrders:
			orderList_c = order_convert(eachOrder)
			totalLength, totalPath, fetchpoints = default_Order(list(orderList_c), edges, start, end)
			if(choice2):
				totalLength2, totalPath2, fetchpoints2, newOrder = opti_Order(list(orderList_c), list(eachOrder), edges, start, end)
			else:
				m, positions = matrixGen(list(orderList_c), edges, start)
				orderSet = set(range(len(positions)))
				ma, cost = reduceMatrix(numpy.array(m))
				st = matrixNode(0, ma, [], [], cost)
				newOrder = bnb(m, orderSet, st, eachOrder)
				totalLength2, totalPath2, fetchpoints2 = default_Order(order_convert(newOrder), edges, start, end)
			
			print("1 order processed")
			print("Here is the optimal picking order:")
			print(newOrder)
			print("Here is the original picking order:")
			print(eachOrder)
			print("Here is the optimal path:")
			sysPrint(list(totalPath2), list(fetchpoints2), newOrder)
			print("Here is the original path:")
			sysPrint(totalPath, fetchpoints, eachOrder)
			print("Here is the optimal path length:")
			print(totalLength2)
			print("Here is the original path length:")
			print(totalLength)

			print("Getting total effort...")
			currentEff = 0
			currentWei = 0
			currentPosition = start
			for each in newOrder:
				if each in item_info:
					newPosition = items_list[each]
					l, p, pt = shortestPath(edges, currentPosition, newPosition)
					currentEff += currentWei * l
					currentWei += item_info[each][3]
					currentPosition = pt
				else:
					print("This item has no dimension infomation:", each)
			l, p = getPath(edges, currentPosition, end)
			currentEff += l * currentWei
			print("Here is the total effort:", currentEff)
			sp_time = time.time() * 1000.0
			print("Time used for this run (ms) = ", sp_time - st_time)
			if(splited):
				print("To see the route for next splited order part, close the plot window")
				print("==========================================================================================")
				print()
			genPlot(totalPath2[0], totalPath2[len(totalPath2) - 1], totalPath2[1: len(totalPath2) - 1], fetchpoints2, orderList_c)

	else:
		print("Please list file of orders to be processed:")
		OrderFile = input()
		print("Please list output file:")
		OutputFile = input()
		OrderFile = open(OrderFile, "r")
		OutputFile = open(OutputFile, "w")
		BatchFile = open("BatchFile.txt", "w")
		print("Computing...")
		st_time = time.time() * 1000.0
		orders = []
		for line in OrderFile:
			line = line[0:len(line)-1]
			get = re.split(r'\t+', line.rstrip('\t'))
			orderList = []
			for each in get:
				orderList.append(int(each))
			orders.append(orderList)

		count = 0

		# comment this to disable order spliting and merging
		orders = splitNmerge(orders)

		for eachOrder in orders:
			orderList_c = order_convert(eachOrder)
			if orderList_c == []:
				continue
				
			count += 1
			OutputFile.write("##Order Number##\n")
			OutputFile.write(str(count) + '\n')
			OutputFile.write("##Worker Start Location##\n")
			OutputFile.write("("+ str(stx)+ ", "+ str(sty)+ ")\n")
			OutputFile.write("## Worker End Location##\n")
			OutputFile.write("("+ str(dex)+ ", "+ str(dey)+ ")\n")

			totalLength, totalPath, fetchpoints = default_Order(list(orderList_c), edges, start, end)
			if(choice2):
				totalLength2, totalPath2, fetchpoints2, newOrder = opti_Order(list(orderList_c), list(eachOrder), edges, start, end)
			else:	
				m, positions = matrixGen(list(orderList_c), edges, start)
				orderSet = set(range(len(positions)))
				ma, cost = reduceMatrix(numpy.array(m))
				st = matrixNode(0, ma, [], [], cost)
				newOrder = bnb(m, orderSet, st, eachOrder)
				totalLength2, totalPath2, fetchpoints2 = default_Order(order_convert(newOrder), edges, start, end)
			OutputFile.write("##Original Parts Order##\n")
			OutputFile.write(str(eachOrder)+'\n')
			OutputFile.write("##Optimized Parts Order##\n")
			OutputFile.write(str(newOrder)+'\n')
			OutputFile.write("##Original Parts Total Distance##\n")
			OutputFile.write(str(totalLength)+'\n')
			OutputFile.write("##Optimized Parts Total Distance##\n")
			OutputFile.write(str(totalLength2)+'\n')
			OutputFile.write("##Optimized Path##\n")
			filePrint(OutputFile, list(totalPath2), list(fetchpoints2), newOrder)
			OutputFile.write("##Getting total effort##\n")
			currentEff = 0
			currentWei = 0
			currentPosition = start
			for each in newOrder:
				if each in item_info:
					newPosition = items_list[each]
					l, p, pt = shortestPath(edges, currentPosition, newPosition)
					currentEff += currentWei * l
					currentWei += item_info[each][3]
					currentPosition = pt
				else:
					OutputFile.write("This item has no dimension infomation:" + str(each) + ".\n")
			l, p = getPath(edges, currentPosition, end)
			currentEff += l * currentWei
			OutputFile.write("Final effort=" + str(currentEff) + "\n")
			OutputFile.write("\n")
			batchWrite(totalPath2[0], totalPath2[len(totalPath2) - 1], totalPath2[1: len(totalPath2) - 1], fetchpoints2, orderList_c, BatchFile, count, currentEff, newOrder)
		OutputFile.close()
		BatchFile.close()
		sp_time = time.time() * 1000.0
		print(count, "orders processed")
		print("Time used for this write (ms) = ", sp_time - st_time)