from heapq import heappush, heappop
import numpy as np


def run_dijkstras(prob_map, start_ind, graph_structure):
    """Run dijkstra's algorithm using graph formed by a specified probability map (weights) and graph structure (edges).
        Start index indicates which node to begin from
        _________

        prob_map: probability map, numpy array with shape: (width, height).
        _________

        graph_structure: a 2D list where the first dimension is the index of the vertex and
        the second dimension contains the indices for the neighbours which it connects to with a directed edge
        _________

        start_ind: start index expects a number indicating what node (index) to start the search from
        _________

        Returns:

            List of shortest paths. One for each vertex indexed by vertex number which is either:
            (1): 0 indicating unreachable vertex or
            (2): a tuple of the form: (final shortest path length, previous vertex index)
        __________

        Indices are calculated by: index = column + (row * graph width)
        __________
        """

    max_ind = prob_map.shape[0] * prob_map.shape[1] - 1

    # setup list containing final shortest paths
    # each element corresponds to a vertex (vertex index = list index) and is a tuple expected to be of the form:
    # (final shortest distance to vertex, previous vertex visited on final shortest path)
    shortest_paths = [None] * (prob_map.shape[0] * prob_map.shape[1])

    # setup queue to contain incomplete vertices
    # each queue entry corresponds to a vertex and is a tuple expected to be of the form:
    # (current shortest distance to vertex, neighbour priority (lower is higher),
    # index of vertex, previous vertex visited on shortest path)
    candidates_q = [(0, 0, 0, start_ind, 0)]

    add_count = 1

    while candidates_q:    # while we still have incomplete vertices
        path_len, _, _, v, a = heappop(candidates_q)
        if shortest_paths[v] is None:
            # we have not found the shortest path for v yet
            shortest_paths[v] = (path_len, a)    # found final shortest path for this vertex

            if v == max_ind:
                # we have found the shortest path to the bottom-rightmost corner -> we are DONE!
                break

            num_neighbours = len(graph_structure[v])
            for i in range(num_neighbours):
                n = graph_structure[v][i]
                cur_v_col = int(v % prob_map.shape[0])
                cur_v_row = int(v / prob_map.shape[0])
                cur_v_prob = prob_map[cur_v_col][cur_v_row]

                neigh_v_col = int(n % prob_map.shape[0])
                neigh_v_row = int(n / prob_map.shape[0])
                neigh_v_prob = prob_map[neigh_v_col][neigh_v_row]

                if cur_v_col == neigh_v_col:
                    # edge_len = 0
                    edge_len = np.max(2 - (cur_v_prob + neigh_v_prob), 0)
                else:
                    edge_len = np.max(2 - (cur_v_prob + neigh_v_prob), 0)

                if shortest_paths[n] is None:
                    # we have not found the shortest path for n yet
                    if neigh_v_col == cur_v_col and neigh_v_row == cur_v_row + 1:
                        heappush(candidates_q, (path_len + edge_len, 0, add_count, n, v))
                    else:
                        heappush(candidates_q, (path_len + edge_len, i + 1, add_count, n, v))

                    add_count += 1
                else:
                    pass
                    # we have already finished this neighbour vertex
                    # we won't find a shorter path so don't bother with it
        else:
            # we have already finished this vertex, we won't find a shorter path so don't bother with it
            pass

    # assign zero distance to unreachable vertices
    return [0 if x is None else x for x in shortest_paths]


def create_graph_structure(shape, max_grad=1):
    """Create structure for a gridded graph of specified shape
        _________

        shape: Shape of graph which is expected as a tuple of the form (width, height).
        Here width does not include the first and last additional appended columns, these are automatically
        appended in here.
        _________

        Returns:

            graph structure info as a 2D list where the first dimension is the index of the vertex
            and the second dimension contains the indices of it's neighbours which it connects to with a directed edge
        _________

        Indices are calculated by: index = column + (row * graph width)
        _________
        """

    graph_width = shape[0] + 2      # append the two extra columns
    graph_height = shape[1]

    # setup graph neighbours as an empty 2D list
    graph = [[] for _ in range(graph_width * graph_height)]

    # for each vertex in the gridded graph
    for i in range(graph_height):
        for j in range(graph_width):

            # calculate indices of various neighbours
            node = j + i * graph_width

            nodes_diagup = []
            nodes_diagdown = []
            for grad in range(1, max_grad + 1):
                node_diagup = (j + 1) + (i - grad) * graph_width
                nodes_diagup.append(node_diagup)

                node_diagdown = (j + 1) + (i + grad) * graph_width
                nodes_diagdown.append(node_diagdown)

            node_right = (j + 1) + i * graph_width
            node_down = (j) + (i + 1) * graph_width

            # add neighbours as required for various cases of first/middle/last, row/col
            # ensuring that non-existent neighbours are not added
            # example non-existent neighbour: a vertex below the bottommost row

            if i == graph_height - 1:
                # last row
                if j == graph_width - 1:
                    # last column
                    pass
                else:
                    # first and middle columns
                    graph[node].append(node_right)
                    for grad_ind in range(0, max_grad):
                        if i - grad_ind - 1 >= 0:
                            # check that we do not overflow above
                            graph[node].append(nodes_diagup[grad_ind])

            elif i == 0:
                # first row
                if j == graph_width - 1:
                    # last column
                    graph[node].append(node_down)
                elif j == 0:
                    # first column
                    graph[node].append(node_right)
                    graph[node].append(node_down)
                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])
                else:
                    # middle column
                    graph[node].append(node_right)
                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])
            else:
                # middle row
                if j == graph_width - 1:
                    # last column
                    graph[node].append(node_down)
                elif j == 0:
                    # first column
                    graph[node].append(node_right)
                    graph[node].append(node_down)

                    for grad_ind in range(0, max_grad):
                        if i - grad_ind - 1 >= 0:
                            # check that we do not overflow above
                            graph[node].append(nodes_diagup[grad_ind])

                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])
                else:
                    # middle column
                    graph[node].append(node_right)

                    for grad_ind in range(0, max_grad):
                        if i - grad_ind - 1 >= 0:
                            # check that we do not overflow above
                            graph[node].append(nodes_diagup[grad_ind])

                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])

    return graph


def create_graph_structure_vertical(shape):
    max_grad = 1

    graph_width = shape[0] + 2      # append the two extra columns
    graph_height = shape[1]

    # setup graph neighbours as an empty 2D list
    graph = [[] for _ in range(graph_width * graph_height)]

    # for each vertex in the gridded graph
    for i in range(graph_height):
        for j in range(graph_width):

            # calculate indices of various neighbours
            node = j + i * graph_width

            nodes_diagup = []
            nodes_diagdown = []
            for grad in range(1, max_grad + 1):
                node_diagup = (j + 1) + (i - grad) * graph_width
                nodes_diagup.append(node_diagup)

                node_diagdown = (j + 1) + (i + grad) * graph_width
                nodes_diagdown.append(node_diagdown)

            node_right = (j + 1) + i * graph_width
            node_down = (j) + (i + 1) * graph_width
            node_up = (j) + (i - 1) * graph_width

            # add neighbours as required for various cases of first/middle/last, row/col
            # ensuring that non-existent neighbours are not added
            # example non-existent neighbour: a vertex below the bottommost row

            if i == graph_height - 1:
                # last row
                if j == graph_width - 1:
                    # last column
                    pass
                else:
                    # first and middle columns
                    graph[node].append(node_right)
                    graph[node].append(node_up)

                    for grad_ind in range(0, max_grad):
                        if i - grad_ind - 1 >= 0:
                            # check that we do not overflow above
                            graph[node].append(nodes_diagup[grad_ind])
            elif i == 0:
                # first row
                if j == graph_width - 1:
                    # last column
                    graph[node].append(node_down)
                elif j == 0:
                    # first column
                    graph[node].append(node_right)
                    graph[node].append(node_down)

                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])
                else:
                    # middle column
                    graph[node].append(node_right)
                    graph[node].append(node_down)

                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])
            else:
                # middle row
                if j == graph_width - 1:
                    # last column
                    graph[node].append(node_down)
                elif j == 0:
                    # first column
                    graph[node].append(node_right)
                    graph[node].append(node_down)

                    for grad_ind in range(0, max_grad):
                        if i - grad_ind - 1 >= 0:
                            # check that we do not overflow above
                            graph[node].append(nodes_diagup[grad_ind])

                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])
                else:
                    # middle column
                    graph[node].append(node_right)
                    graph[node].append(node_up)
                    graph[node].append(node_down)

                    for grad_ind in range(0, max_grad):
                        if i - grad_ind - 1 >= 0:
                            # check that we do not overflow above
                            graph[node].append(nodes_diagup[grad_ind])

                    for grad_ind in range(0, max_grad):
                        if i + grad_ind + 1 <= graph_height - 1:
                            # check that we do not overflow below
                            graph[node].append(nodes_diagdown[grad_ind])

    return graph


def append_firstlast_cols(prob_map):
    """Append first and last columns of probability one (maximum probability) to a given map
        _________

        prob_map: probability map, numpy array with shape: (width, height)
        Values in map should be normalised between 0 and 1.
        _________

        Returns the modified map with appended columns. shape: (width + 2, height)
        _________
        """

    map_height = prob_map.shape[1]

    prob_map = np.concatenate((np.ones((1, map_height)), prob_map), axis=0)  # append first col
    prob_map = np.concatenate((prob_map, np.ones((1, map_height))), axis=0)  # append last col

    return prob_map


def delineate_boundary(prob_map, graph_structure):
    """Delineate boundary (obtain a single row prediction for each column) for given probability map using a
        gridded graph constructed both from the probabilities of the map and the specified graph connectivity structure.
        _________

        prob_map: probability map, numpy array with shape (width, height).
        Values in map should be normalised between 0 and 1.
        _________

        graph_structure: a 2D list where the first dimension is the index of the vertex and
        the second dimension contains the indices for the neighbours connected with a directed edge
        _________

        Returns a numpy array containing the delineated boundary positions for the prob map (one row position
        for each column as required), shape: (width,)
        _________

        Indices are calculated by:

            index = column + row * graph width

        To go back and extract row and column for position tuple of the form (col, row):

            (ind % width, ind / width) = (ind % width, floor(ind / width))
        _________
        """

    prob_map = append_firstlast_cols(prob_map)
    shortest_paths = run_dijkstras(prob_map, 0, graph_structure)

    map_width = prob_map.shape[0]
    map_height = prob_map.shape[1]

    # extract shortest path starting at bottom right corner and working back

    final_node_ind = (map_width * map_height) - 1

    node_ind = final_node_ind  # current node index
    node_coord = (node_ind % map_width, int(node_ind / map_width))  # current node coordinate
    prev_node_ind = shortest_paths[node_ind][1]  # previous node index

    node_order_coords = []  # list of node coordinates along shortest path in reverse order

    while node_coord != (0, 0):  # keep adding while we haven't reached the start vertex
        node_order_coords.append(node_coord)
        next_node_coord = (prev_node_ind % map_width, int(prev_node_ind / map_width))
        node_coord = next_node_coord
        prev_node_ind = shortest_paths[prev_node_ind][1]

    delin = np.zeros((map_width - 2))    # numpy array of row values corresponding to the delineated boundary
    # (one for each column in the original map: exclude the appended columns)

    for coord in node_order_coords:
        # do not add the coordinate if it is part of the first or last column
        # these first and last columns do not form part of the delineation
        if coord[0] != 0 and coord[0] != map_width - 1:
            delin[coord[0] - 1] = coord[1]

    return delin


def delineate_boundary_vertical(prob_map, graph_structure):
    prob_map = append_firstlast_cols(prob_map)
    shortest_paths = run_dijkstras(prob_map, 0, graph_structure)

    map_width = prob_map.shape[0]
    map_height = prob_map.shape[1]

    # extract shortest path starting at bottom right corner and working back

    final_node_ind = (map_width * map_height) - 1

    node_ind = final_node_ind  # current node index
    node_coord = (node_ind % map_width, int(node_ind / map_width))  # current node coordinate
    prev_node_ind = shortest_paths[node_ind][1]  # previous node index

    node_order_coords = []  # list of node coordinates along shortest path in reverse order

    while node_coord != (0, 0):  # keep adding while we haven't reached the start vertex
        node_order_coords.append(node_coord)
        next_node_coord = (prev_node_ind % map_width, int(prev_node_ind / map_width))
        node_coord = next_node_coord
        prev_node_ind = shortest_paths[prev_node_ind][1]

    delin = np.zeros((map_width - 2))    # numpy array of row values corresponding to the delineated boundary
    counts = np.zeros((map_width - 2))
    # (one for each column in the original map: exclude the appended columns)

    for coord in node_order_coords:
        # do not add the coordinate if it is part of the first or last column
        # these first and last columns do not form part of the delineation
        if coord[0] != 0 and coord[0] != map_width - 1:
            delin[coord[0] - 1] += coord[1]
            counts[coord[0] - 1] += 1

    for col in range(delin.shape[0]):
        delin[col] = delin[col] / counts[col]

    return delin


def calc_errors(prediction, truth):
    """Calculate delineation errors by comparing the predictions and truths.
        Predictions or truths that are NaN or <= 0 have a NaN error.
        Predictions and truths must be the same shape.

        Errors are calculated by:

        error = predicted value - true value
        _________

        prediction: numpy array of integer values corresponding to the row prediction for each column.
        shape: (width,)
        _________

        truth: numpy array of integer values corresponding to the true row position for each column.
        shape: (width,)
        _________

        Returns numpy array containing the errors. Shape: (width,). Where error cannot be calculated or is invalid,
        it is replaced by np.nan
        _________
        """

    width = prediction.shape[0]
    error = np.zeros((width,), dtype='float64')

    for i in range(width):
        if np.isnan(truth[i]):
            error[i] = np.nan
        elif truth[i] <= 0:
            error[i] = np.nan
        else:
            error[i] = prediction[i].astype('float64') - truth[i]

    return error


def segment_maps(prob_maps, truths, eval_params=None, graph_structure=None):
    """Delineate boundaries using specified neighbours structure for a number of probability maps
        and subsequently calculate delineation errors.
        _________

        prob_maps: numpy array of probability maps with the shape: (number of maps/boundaries, width, height).
        Probability map values assumed to be uint8 between 0 and 255. These will be normalised to between float64
        0 and 1 here.
        _________

        truths: numpy array of values with the shape: (number of maps/boundaries, width) corresponding to the true row
        locations for each column for each map.
        _________

        graph_structure: a 2D list where the first dimension is the index of each vertex and
        the second dimension contains the indices for the neighbours connected with a directed edge
        _________

        Returns delineations and errors for each probability map in numpy arrays. Two structures:

        (1) predictions: numpy array with shape: (number of maps, width) corresponding with a
        predicted value for each column for each map

        (2) errors: numpy array with shape: (number of maps, width) corresponding to the error between the predicted
        and true value for each column for each map
        _________
        """

    if eval_params is not None:
        graph_structure = eval_params.graph_structure

    prob_maps.astype('float64')
    prob_maps = prob_maps / 255

    num_maps = prob_maps.shape[0]
    width = prob_maps.shape[1]

    predictions = np.zeros((num_maps, width), dtype='uint16')
    errors = np.zeros((num_maps, width), dtype='float64')

    if eval_params is not None and eval_params.trim_maps is True:
        map_ind = eval_params.trim_ref_ind
        if eval_params.vertical_graph_search is False:
            ref_prediction = delineate_boundary(prob_maps[map_ind], graph_structure)
        elif eval_params.vertical_graph_search is True:
            ref_prediction = delineate_boundary_vertical(prob_maps[map_ind], graph_structure)
        elif eval_params.vertical_graph_search == "ilm_vertical":
            if map_ind == 0:
                ref_prediction = delineate_boundary_vertical(prob_maps[map_ind], graph_structure[0])
            else:
                ref_prediction = delineate_boundary(prob_maps[map_ind], graph_structure[1])

        predictions[map_ind, :] = ref_prediction

        if eval_params.flatten_pred_edges is True:
            predictions[map_ind, :eval_params.flat_marg] = predictions[map_ind, eval_params.flat_marg]
            predictions[map_ind, -eval_params.flat_marg:] = predictions[map_ind, -eval_params.flat_marg]

        if truths is not None:
            error = calc_errors(ref_prediction, truths[map_ind, :])
            errors[map_ind, :] = error

        top_bounds = ref_prediction.astype('uint16') - eval_params.trim_window[0]
        bottom_bounds = ref_prediction.astype('uint16') + eval_params.trim_window[1]

        top_bounds[top_bounds > 1000] = 0
        bottom_bounds[bottom_bounds > 1000] = 0

        for map_ind in range(num_maps):
            if map_ind == eval_params.trim_ref_ind:
                continue

            for col in range(prob_maps.shape[1]):
                prob_maps[map_ind, col, 0:top_bounds[col]] = 0
                prob_maps[map_ind, col, bottom_bounds[col]:] = 0

            if eval_params.vertical_graph_search is False:
                prediction = delineate_boundary(prob_maps[map_ind], graph_structure)
            elif eval_params.vertical_graph_search is True:
                prediction = delineate_boundary_vertical(prob_maps[map_ind], graph_structure)
            elif eval_params.vertical_graph_search == "ilm_vertical":
                if map_ind == 0:
                    prediction = delineate_boundary_vertical(prob_maps[map_ind], graph_structure[0])
                else:
                    prediction = delineate_boundary(prob_maps[map_ind], graph_structure[1])

            predictions[map_ind, :] = prediction

            if eval_params.flatten_pred_edges is True:
                predictions[map_ind, :eval_params.flat_marg] = predictions[map_ind, eval_params.flat_marg]
                predictions[map_ind, -eval_params.flat_marg:] = predictions[map_ind, -eval_params.flat_marg]

            if truths is not None:
                error = calc_errors(prediction, truths[map_ind, :])
                errors[map_ind, :] = error
    else:
        for map_ind in range(num_maps):
            if eval_params.vertical_graph_search is False:
                prediction = delineate_boundary(prob_maps[map_ind], graph_structure)
            elif eval_params.vertical_graph_search is True:
                prediction = delineate_boundary_vertical(prob_maps[map_ind], graph_structure)
            elif eval_params.vertical_graph_search == "ilm_vertical":
                if map_ind == 0:
                    prediction = delineate_boundary_vertical(prob_maps[map_ind], graph_structure[0])
                else:
                    prediction = delineate_boundary(prob_maps[map_ind], graph_structure[1])

            predictions[map_ind, :] = prediction

            if eval_params.flatten_pred_edges is True:
                predictions[map_ind, :eval_params.flat_marg] = predictions[map_ind, eval_params.flat_marg]
                predictions[map_ind, -eval_params.flat_marg:] = predictions[map_ind, -eval_params.flat_marg]

            if truths is not None:
                error = calc_errors(prediction, truths[map_ind, :])
                errors[map_ind, :] = error

    return predictions, errors, prob_maps


def calculate_overall_errors(errors, col_error_range):
    num_boundaries = errors.shape[0]

    mean_abs_err = np.zeros((num_boundaries, ), dtype='float64')
    mean_err = np.zeros((num_boundaries, ), dtype='float64')
    abs_err_sd = np.zeros((num_boundaries, ), dtype='float64')
    err_sd = np.zeros((num_boundaries, ), dtype='float64')

    errors = errors[:, col_error_range[0]:col_error_range[-1] + 1]

    for boundary_ind in range(num_boundaries):
        mean_abs_err[boundary_ind] = np.nanmean(np.abs(errors[boundary_ind]))
        mean_err[boundary_ind] = np.nanmean(errors[boundary_ind])
        abs_err_sd[boundary_ind] = np.nanstd(np.abs(errors[boundary_ind]))
        err_sd[boundary_ind] = np.nanstd(errors[boundary_ind])

    return [mean_abs_err, mean_err, abs_err_sd, err_sd]

