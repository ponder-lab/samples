#!/usr/bin/env python

import random
import copy
import graphviz

class operators():
    def __init__(self, ops=[], arities=[]):
        self.ops = ops.copy()
        self.arities = arities.copy()

    def add_ops(self, op, arity):
        self.ops += [op]
        self.arities += [arity]

    def max_arity(self):
        return max(self.arities)

    def num(self):
        return len(self.ops)

class genotype(list):
    # Convert row/col index to node address
    def get_node_address(self, j, i):
        return self.num_input + i*self.rows + j

    # Get accessible node index range according to levels-back parameter
    def get_accessible_range(self, i):
        head_col = i - self.levels_back
        head_address = 0 if head_col < 0 else self.get_node_address(0, head_col)
        return range(head_address, self.get_node_address(0, i))
    
    def __init__(self, num_input, num_output, rows, cols, levels_back, ops):

        self.num_input = num_input
        self.num_output = num_output
        self.rows = rows
        self.cols = cols
        self.levels_back = levels_back

        num_ops = ops.num()
        max_arity = ops.max_arity()
        self.node_len = max_arity + 1

        # Set initial random genes for each operation node
        for i in range(cols):
            accessible_range = self.get_accessible_range(i)
            for j in range(rows):
                for a in range(max_arity):
                    self += [random.randrange(accessible_range.start,
                                              accessible_range.stop)]
                self += [random.randrange(num_ops)]

        # Set initial random genes for each output node
        accessible_range = self.get_accessible_range(cols)
        for i in range(num_output):
            self += [random.randrange(accessible_range.start,
                                      accessible_range.stop)]

    def copy(self):
        return copy.copy(self) #!!!!!!!! deepcopy if 2D array !!!!!!!!

    def get_valid_node(self, ops):
        used_address = [False] * (self.num_input + self.rows*self.cols)

        # referenced by output node
        for i in range(self.num_output):
            used_address[self[-1-i]] = True

        # referenced by function node
        for i in range(self.cols):
            for j in range(self.rows):
                idx = i * self.rows * self.node_len + \
                      j * self.node_len
                op_idx = self[idx + self.node_len - 1]
                for a in range(ops.arities[op_idx]):
                    used_address[self[idx+a]] = True

        valid_node = []
        for i in range(self.num_input, len(used_address)):
            if used_address[i]:
                valid_node += [i]

        return valid_node

    def decode(self, idata, ops):
        data = [0] * (self.num_input + self.rows*self.cols)

        for i in range(self.num_input):
            data[i] = idata[i]

        valid_node = self.get_valid_node(ops)
        for node_idx in valid_node:
            gene_idx = (node_idx - self.num_input) * self.node_len
            op_idx = self[gene_idx + self.node_len - 1]
            arity = ops.arities[op_idx]
            node_input = [0] * arity
            for j in range(arity):
                node_input[j] = data[self[gene_idx + j]]
            op = ops.ops[op_idx]
            data[node_idx] = op(*node_input)

        odata = [0] * self.num_output
        for i in range(self.num_output):
            odata[i] = data[self[-self.num_output+i]]

        return odata

    def mutate(self, num_mutate, ops):
        gene_idx_list = []
        while len(gene_idx_list) < num_mutate:
            idx = random.randrange(len(self))
            if idx not in gene_idx_list:
                col = idx // (self.rows * self.node_len)
                accessible_range = self.get_accessible_range(col)

                # Output node
                if idx >= len(self) - self.num_output:
                    self[idx] = random.randrange(accessible_range.start,
                                                 accessible_range.stop)
                    gene_idx_list += [idx]
                # Operation node
                else:
                    sub_idx = idx % self.node_len
                    op_idx = self[idx - sub_idx + self.node_len - 1]
                    arity = ops.arities[op_idx]

                    # Operation type node
                    if sub_idx == self.node_len - 1:
                        num_ops = ops.num()
                        new_op = random.randrange(num_ops)
                        new_arity = ops.arities[new_op]
                        old_arity = ops.arities[self[idx]]
                        # if # of arity increase assign unconnected input port
                        if new_arity > old_arity:
                            node_origin = idx + 1 - self.node_len
                            for i in range(old_arity, new_arity):
                                self[node_origin+i] = random.randrange(accessible_range.start,
                                                                       accessible_range.stop)
                        self[idx] = random.randrange(num_ops)
                        gene_idx_list += [idx]
                    # Operation input node
                    elif sub_idx < arity:
                        self[idx] = random.randrange(accessible_range.start,
                                                 accessible_range.stop)
                        gene_idx_list += [idx]
                    # Invalid gene
                    #else:

    def draw_graph(self, ops, name="test"):
        graph = graphviz.Digraph(name, format="png")
        for i in range(self.num_input):
            graph.node("in%d" % i)

        for i in range(self.num_output):
            graph.node("out%d" % i)

        for i in range(self.cols):
            for j in range(self.rows):
                idx = i * self.rows * self.node_len + \
                      j * self.node_len
                op_idx = self[idx + self.node_len - 1]

                node_name = "op" + str(j) + "," + str(i)
                node_label = ops.ops[op_idx].__name__
                graph.node(node_name, node_label)
        
                for a in range(ops.arities[op_idx]):
                    addr = self[idx+a]
                    if addr < self.num_input:
                        graph.edge("in%d" % addr, node_name)
                    else:
                        src_i = (addr - self.num_input) // self.rows
                        src_j = (addr - self.num_input) % self.rows
                        src_name = "op" + str(src_j) + "," + str(src_i)
                        graph.edge(src_name, node_name)

        for i in range(self.num_output):
            idx = self.rows * self.cols * self.node_len + i
            addr = self[idx]
            if addr < self.num_input:
                graph.edge("in%d" % addr, "out%d" % i)
            else:
                src_i = (addr - self.num_input) // self.rows
                src_j = (addr - self.num_input) % self.rows
                src_name = "op" + str(src_j) + "," + str(src_i)
                graph.edge(src_name, "out%d" % i)

        graph.view()

# Compute total fitness
def evaluate_genotype(genotype, idata_list, expdata_list, ops, func_fitness):
    sum_fitness = 0.0
    for idata, expdata in zip(idata_list, expdata_list):
        odata = genotype.decode(idata, ops)
        fitness = func_fitness(expdata, odata)
        #print("in =", idata, "out =", odata, "exp =", expdata, "fitness =", fitness)
        sum_fitness += fitness
    #print("sum_fitness =", sum_fitness)
    return sum_fitness

# Select best genotype in the current group
def find_best_genotype(genotypes, idata_list, expdata_list, ops, func_fitness):
    max_fitness = -1.0e10
    for i, genotype in enumerate(genotypes):
        fitness = evaluate_genotype(genotype, idata_list, expdata_list,
                                    ops, func_fitness)
        if fitness >= max_fitness:
            max_fitness = fitness
            best_idx = i
    return max_fitness, best_idx

