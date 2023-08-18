import torch
import torch.nn as nn

from collections import Counter


class NAVQ(nn.Module):

    def __init__(self, num_classes=100, feat_dim=2, device=None, with_grow=1):
        super(NAVQ, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.cvs = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        self.edges = torch.zeros([self.num_classes, self.num_classes]).to(self.device)
        self.edges.fill_diagonal_(1)
        self.print_edges_counter = 0

        self.cv_class = dict(zip(range(self.num_classes), [i for i in range(self.num_classes)]))
        self.class_indices = [[i] for i in range(self.num_classes)]
        self.num_cvs = self.num_classes

        self.cv_connectedness = torch.zeros([self.num_classes, self.num_classes]).to(self.device)
        self.cv_connectedness.fill_diagonal_(1)

    def add_cvs(self, num_classes_to_add):

        new_cvs = torch.randn(num_classes_to_add, self.feat_dim, device=self.device)
        self.cvs = nn.Parameter(
            torch.cat((self.cvs,
                       new_cvs)).to(
                self.device))

        for i in range(num_classes_to_add):
            self.cv_class.update({self.num_cvs + i: self.num_classes + i})
            self.class_indices.append([self.num_cvs + i])

        self.num_classes += num_classes_to_add
        self.num_cvs += num_classes_to_add

        self.optimizer.add_param_group({"params": new_cvs})

        edges_new = torch.zeros([self.num_cvs, self.num_cvs])
        edges_new[:self.num_cvs - num_classes_to_add, :self.num_cvs - num_classes_to_add] = self.edges
        self.edges = edges_new.to(self.device)
        self.edges.fill_diagonal_(1)


        cv_connectedness_new = torch.zeros([self.num_cvs, self.num_cvs])
        cv_connectedness_new[:self.num_cvs - num_classes_to_add,
        :self.num_cvs - num_classes_to_add] = self.cv_connectedness
        self.cv_connectedness = cv_connectedness_new.to(self.device)
        self.cv_connectedness.fill_diagonal_(1)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # TODO move k,epsilon
        self.k = 15

        # The edge-strength decay constant. Set at a small value for sparse graph generation. For dense
        # graphs, use a higher value (close to 1.0 but less than 1.0).

        epsilon = 0.9
        e_min = 0.9 ** 10
        self.print_edges_counter += 1

        dist_to_cvs = torch.cdist(x, self.cvs)
        sorted_dists, ordered_cvs = torch.sort(dist_to_cvs, dim=1)

        correct_cvs = torch.tensor(labels)

        kth_closest_cvs = ordered_cvs[:, :self.k]

        # # ------------------------------------------------------------------------------------------------------------

        closest_cvs_list = kth_closest_cvs[:, 0].tolist()
        # counting the number of times the closest cvs appear in the input, encounters is the number of time by
        # which we multiply the edge strength in the input by epsilon in the non optimised code
        visited_node_encounters = torch.zeros(self.num_cvs, device=self.device)
        visits_counter = Counter(closest_cvs_list)
        for i in range(self.num_cvs):
            if i in visits_counter:
                visited_node_encounters[i] = visits_counter[i]
        encounters = (torch.ones(self.num_cvs, self.num_cvs, device=self.device) * visited_node_encounters).T

        # closest cvs tensor gives the number of times each cv was considered as a closest cv of another cv.
        # this is the condition where we set the edge strength to 1 in the non-optimised code
        closest_cv_encounters = torch.zeros(self.num_cvs, self.num_cvs, device=self.device)
        for i, i_k in enumerate(kth_closest_cvs.tolist()):
            for cv in i_k:
                closest_cv_encounters[closest_cvs_list[i], cv] += 1

        epsilon = epsilon * torch.ones(self.num_cvs, self.num_cvs, device=self.device)

        # this array is used to set the start of the edge strength to either
        #   1 (if the edge is considered as a closest neighbour connection in this iteration) or
        #   current value (if the edge is not considered as closest neighbour connection in this iteration)
        #   0 (if there has been no updates on this edge so far).

        # this identify whether its an edge that should updated in this iteration
        closest_cvs_existence = torch.gt(closest_cv_encounters, 0)

        # recalculate encounters by subtracting the closest cv occurrences to identify the number by which the edge
        # strength should be multiplied by epsilon
        # here we ignore the order of the operations and assume
        # if encounters> closest_cv_encounters:
        #   multiply edge strength  by epsilon**(encounters-closest_cv_encounters) times
        # else
        #   edge strength=1
        encounters = (encounters - closest_cv_encounters) * (torch.gt((encounters - closest_cv_encounters), 0))

        # if first_time: self.edges = closest_cvs_existence since we're setting the edge strength to 1 at
        # closest_cvs_existence we dont need to update the value to 1 again when encounters<closest cv
        # encounters
        self.edges = torch.max(self.edges, closest_cvs_existence)
        self.edges = self.edges * (epsilon ** encounters)
        self.edges = self.edges * (1 - (self.edges < e_min) * 1)

        scale_factor = torch.diagonal(
            torch.pow(torch.cdist(x, torch.index_select(self.cvs, 0, correct_cvs)), 2), 0)

        x_cvs = torch.pow(dist_to_cvs, 2)

        d_pos = scale_factor

        d_neg = (x_cvs * torch.index_select(
            torch.logical_and((self.edges > 0), self.cv_connectedness < 1),
            0,
            correct_cvs))

        exp_d_neg_neighbours = torch.exp(-0.001 * d_neg) * (d_neg > 0)  # d_neg>0 to remove non neighbours
        w_d_neg = exp_d_neg_neighbours / exp_d_neg_neighbours.sum(keepdim=True, dim=1)
        w_d_neg[w_d_neg != w_d_neg] = 0  # removing nan
        w_d_neg = w_d_neg.detach()

        mu = (d_pos - (w_d_neg * d_neg).sum(dim=1))
        loss = (nn.ReLU()(mu)).sum() / x.size(0)

        self.edges = (self.edges + self.edges.T) / 2

        return loss
