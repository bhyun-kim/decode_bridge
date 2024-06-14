import torch
import torch.nn as nn
import torch.nn.functional as F

from operations import get_candidates, FactorizedReduction, ReLUConvBN

def parse_alphas(weights, primitives):
    gene = []
    n = 2
    start = 0
    for i in range(4):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != primitives.index('none')))[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k != primitives.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((primitives[k_best], j))
        start = end
        n += 1

    return gene

class SearchCell(nn.Module):
    def __init__(self,
                 prev_prev_channels,
                 prev_channels,
                 out_channels,
                 stride = 4,
                 peaks=[3, 5],
                 num_nodes=7,
                 reduction=False,
                 following_reduction=False):
        super(SearchCell, self).__init__()

        assert num_nodes > 3 # num of input is 2, and num of output is 1

        self.num_nodes = num_nodes
        self.num_intermediates = self.num_nodes - 3
        self.stride = stride if reduction else 1
        intermediate_channels = out_channels // self.num_intermediates

        self.conv_prev = nn.Sequential(nn.ReLU(),
                                       nn.Conv1d(prev_channels,
                                                 intermediate_channels,
                                                 kernel_size=1,
                                                 bias=False),
                                       nn.BatchNorm1d(intermediate_channels,
                                                      affine=False))

        if following_reduction:
            self.conv_prev_prev = FactorizedReduction(prev_prev_channels,
                                                      intermediate_channels)
        else:
            self.conv_prev_prev = nn.Sequential(nn.ReLU(),
                                                nn.Conv1d(prev_prev_channels,
                                                          intermediate_channels,
                                                          kernel_size=1,
                                                          bias=False),
                                                nn.BatchNorm1d(intermediate_channels,
                                                               affine=False))

        self.edges = nn.ModuleList()

        for i in range(self.num_intermediates):
            for j in range(i+2):
                edge_stride = self.stride if j < 2 else 1
                candidates = get_candidates(peaks,
                                            intermediate_channels,
                                            stride=edge_stride)
                self.edges.append(candidates)

    def forward(self,
                input_prev_prev,
                input_prev,
                alphas):

        predecessors = [self.conv_prev_prev(input_prev_prev), self.conv_prev(input_prev)]

        edge_idx = -1
        for i in range(self.num_intermediates):
            _node = []
            for j in range(i+2):
                edge_idx += 1

                _alpha = alphas[edge_idx, :]
                alpha = F.softmax(_alpha, dim=0)

                edge = self.edges[edge_idx]

                for edge_j, _edge in enumerate(edge):
                    _node.append(alpha[edge_j] * _edge(predecessors[j]))

            node = torch.sum(torch.stack(_node), dim=0)
            predecessors.append(node)

        output = torch.cat(predecessors[2:], dim=1)

        return output
    

class SearchNetwork(nn.Module):
    def __init__(self,
                 num_candidates=10,
                 num_intermediates=4,
                 stride=4,
                 num_cells=8,
                 init_channels=16,
                 peaks=[3, 5],
                 depth_multiplier_at_reduction=2):
        super(SearchNetwork, self).__init__()

        self.num_edges = sum(i+2 for i in range(num_intermediates))
        self.num_candidates = num_candidates
        self.num_cells = num_cells

        self.alphas_normal = torch.randn((self.num_edges, self.num_candidates), requires_grad=True)
        self.alphas_reduce = torch.randn((self.num_edges, self.num_candidates), requires_grad=True)

        self._alphas = [self.alphas_normal, self.alphas_reduce]

        prev_prev_channels = prev_channels = init_channels * 3
        current_channels = init_channels * num_intermediates

        self.conv_stem = nn.Sequential(nn.Conv1d(1, prev_channels, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm1d(prev_channels))

        self.cells = nn.ModuleList()
        following_reduction = False

        self.reduction_idx = [self.num_cells // 3, self.num_cells * 2 // 3]
        for i in range(self.num_cells):
            if i in self.reduction_idx:
                current_channels = current_channels * depth_multiplier_at_reduction

                self.cells.append(SearchCell(prev_prev_channels,
                                       prev_channels,
                                       current_channels,
                                       stride=stride,
                                       peaks=peaks,
                                       reduction=True))

                prev_prev_channels, prev_channels = prev_channels, current_channels
                following_reduction = True

            else:
                self.cells.append(SearchCell(prev_prev_channels,
                                       prev_channels,
                                       current_channels,
                                       stride=stride,
                                       peaks=peaks,
                                       following_reduction=following_reduction))

                prev_prev_channels, prev_channels = prev_channels, current_channels
                following_reduction = False

        self.output = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                    nn.Flatten())

    def forward(self, input):

        prev_prev = prev = self.conv_stem(input)

        for i, cell in enumerate(self.cells):
            if i in self.reduction_idx:
                prev_prev, prev = prev, cell(prev_prev, prev, self.alphas_reduce)
            else:
                prev_prev, prev = prev, cell(prev_prev, prev, self.alphas_normal)

        return self.output(prev)

    def alphas(self):
        return self._alphas
    


class Cell(nn.Module):

    def __init__(self,
                genotype,
                prev_prev_channels,
                prev_channels,
                out_channels,
                stride,
                reduction,
                following_reduction,
                operations
                ):
        super(Cell, self).__init__()

        if following_reduction:
            self.preprocess0 = FactorizedReduction(prev_prev_channels, out_channels)
        else:
            self.preprocess0 = ReLUConvBN(prev_prev_channels, out_channels, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(prev_channels, out_channels, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(out_channels, op_names, stride, indices, concat, reduction, operations)

    def _compile(self,
                out_channels,
                op_names,
                stride,
                indices,
                concat,
                reduction,
                operations
                ):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            _stride = stride if reduction and index < 2 else 1
            op = operations[name](out_channels, _stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]

            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

    

class Network(nn.Module):

    def __init__(self,
                init_channels,
                num_cells,
                stride,
                genotype):
        super(Network, self).__init__()

        stem_multiplier = 3
        current_channels = stem_multiplier*init_channels
        self.stem = nn.Sequential(
        nn.Conv1d(1, current_channels, 3, padding=1, bias=False),
        nn.BatchNorm1d(current_channels)
        )

        prev_prev_channels, prev_channels, current_channels = current_channels, current_channels, init_channels
        self.cells = nn.ModuleList()
        following_reduction = False
        for i in range(num_cells):
            if i in [num_cells//3, 2*num_cells//3]:
                current_channels *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype, prev_prev_channels, prev_channels,
                current_channels, stride, reduction, following_reduction
                )
            following_reduction = reduction
            self.cells += [cell]
            prev_prev_channels, prev_channels = prev_channels, cell.multiplier*current_channels

        self.output = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                    nn.Flatten())

    def forward(self, input):
        prev_prev = prev = self.stem(input)
        for i, cell in enumerate(self.cells):
            prev_prev, prev = prev, cell(prev_prev, prev)

        return self.output(prev)


