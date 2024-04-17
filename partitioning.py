# Contains functions to partition data into clients.
# For convience every function returns a list of (ix, cols) for each client and
# must have random_state as an optional argument.

import numpy as np

MIN_SIZE = 500
MIN_PER_CLASS = 10

def natural_partition(df, natural_col, min_size = MIN_SIZE, random_state = None):
    """Partitions dataframe into groups based on a natural column."""
    cols = df.columns.tolist()
    cols.remove(natural_col)
    out = []
    for val in df[natural_col].unique():
        ix = df[df[natural_col] == val].index
        if len(ix) < min_size:
            continue
        out.append((ix, cols))
    return out


def random_horizontal_partition(df, c_clients, n = None, random_state = None):
    """
    Randomly partitions a into a # c_clients of clients.
    Shuffles np.random.arange(a.size) and slices it according to n.

    Input:
        df: pd.DataFrame to partition.
        c_clients: int denoting the number of clients to partition among.
        n: array or None, optional. Denotes the number of labels to be assigned to each client. Must sum to a.size.

    Returns:
        partition: a dictionary {client:array of indices}.
    """

    # If no size distribution passed assume equal
    if n is None:
        n = (np.ones(c_clients)*(df.shape[0]//c_clients)).astype('int')
    
    elif isinstance(n,list):
        n = np.array(n)
    
    
    # We're only interested in the indices
    ix = np.arange(df.shape[0])
    
    # Shuffle the indices
    np.random.seed(random_state)
    np.random.shuffle(ix)
    
    # The partition to output. List of (ix, cols) for each client.
    partition = []
    
    # Splice shuffled a according to n
    i = 0
    for c in range(c_clients):
        partition.append((ix[i:i+n[c]], df.columns.tolist()))
        i += n[c]
        
    # Shuffle islands IDs
    np.random.shuffle(partition)
    
    return partition

def _power_n(df, c_clients, a, min_size = MIN_SIZE):
    # Generate a power law distribution
    bins = np.linspace(0, 1, c_clients + 1)
    data = np.random.power(a = a,size = len(bins)*10000)
    counts,_ = np.histogram(data,bins)
    p = counts / counts.sum()
    n = (np.ones(shape = p.shape) * min_size).astype('int')
    can_distribute = df.shape[0] - (c_clients * min_size)
    assert can_distribute > c_clients, 'Minimum too big'
    n += (p * can_distribute).astype('int')
    return n


def power_partition_n(df, c_clients, a, min_size = MIN_SIZE, random_state = None):
    """
    Partition on size of n. Will follow power law with
    coeficient a in (0,1]. a = 1 is equivalent to the
    uniform distribution while a --> 0 concentrates the mass
    on a single client.
    """
    n = _power_n(df, c_clients, a, min_size)
    return random_horizontal_partition(df, c_clients, n, random_state)


def _redistribute_cells(matrix, threshold, random_state = None):
    """
    Generated by Chat GPT (May 24 Version) with the following prompt:
        I have a numpy matrix where each row of integers adds to a constant.
        I want cells that are under a threshold to 'take' units from the other
        cells in its row at random so that every cell is over the threshold
        and the rows still add up to the same constant.
        
        Can you give my the python numpy code to do so?
    """
    num_rows, num_cols = matrix.shape
    row_sums = np.sum(matrix, axis = 1)
    deficit_indices = np.where(matrix < threshold)
    np.random.seed(random_state)

    while deficit_indices[0].size > 0:
        row_index, col_index = deficit_indices[0][0], deficit_indices[1][0]
        deficit = threshold - matrix[row_index, col_index]
        non_deficit_indices = np.delete(np.arange(num_cols), col_index)

        if deficit > 0:
            surplus_indices = np.where(matrix[row_index] > threshold)
            surplus_values = matrix[row_index, surplus_indices]
            surplus_sum = np.sum(surplus_values)

            if surplus_sum == 0:
                continue

            redistribution_probabilities = surplus_values / surplus_sum
            redistribution_units = np.random.multinomial(deficit, redistribution_probabilities.flatten())

            matrix[row_index, surplus_indices] -= redistribution_units
            matrix[row_index, col_index] += np.sum(redistribution_units)
        
        deficit_indices = np.where(matrix < threshold)

    return matrix

def _samples_per_class_dirichlet(n_classes, c_clients, alpha, n = None, debug = False, min_per_class = 0.0, random_state = None):
    """
    
    Returns the number of samples the nth client must sample from each class
    according to the Dirichlet distribution with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    Parameters
    ----------
    y : numpy array
        The numpy array of labels to be partitioned, assumed to be of integers 0 to
        # of classes -1.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the *number* of elements of y that the i-th client must sample.

    debug : boolean, optional
        Whether to perform extra checks (which can be slow) 
    
    Returns
    -------
    A numpy array of shape(c,k) matrix where A[i,j] denotes
    the amount of instances of class j the client i must draw.

    """
    assert alpha > 0

    np.random.seed(random_state)
    
    # Sample from Dirichelts Dist.
    # proportions[i][j] indicates the proportion of class j that client i must draw
    proportions = np.random.dirichlet(alpha * np.ones(n_classes), c_clients)
    
    # Multiply by n and cast as int
    for client,client_i_n in enumerate(n):
        proportions[client,:] *= client_i_n

    out = proportions.astype('int')
    
    # Correct errors caused by truncation
    missing_by_client = n - out.sum(axis=1)
    assert all(missing_by_client >= 0),'Possible overflow'
    for client, n_missed_by_client in enumerate(missing_by_client):
        where_to_add = np.random.choice(n_classes, size = n_missed_by_client)
        np.add.at(out[client,:], where_to_add, 1)
    
    if debug:
        # Total of output must equal total of input
        assert out.sum() == sum(n)

    min_n_per_class = int(out[0,:].sum() * min_per_class)
    return _redistribute_cells(out, min_n_per_class, random_state)

def _dirichlet_partition(df, y, c_clients, alpha, only_with_labels = None, random_state = None, debug = None):

    assert isinstance(c_clients, int) and c_clients > 0
    assert alpha > 0

    # The number of classes if y is assumed to be pandas' categorical codes.
    if only_with_labels is None:
        classes, counts_y = np.unique(y, return_counts = True)
        n_classes = len(counts_y)
    
    else:
        classes, counts_y = np.unique(y[y.isin(only_with_labels)], return_counts = True)
        n_classes = len(classes)

    # Max n such that all alphas can be guaranteed
    # The worst case that can occur is if one client is assigned
    n_max_all_alphas = counts_y.min()

    # All clients sample same number of points
    n = [n_max_all_alphas // c_clients] * c_clients 
    
    # Given how many examples each client must sample from each class
    how_many = _samples_per_class_dirichlet(
        n_classes = n_classes,
        c_clients = c_clients,
        alpha = alpha,
        n = n,
        debug = debug,
        random_state = random_state
    )

    # Assert we have enough instances from each class
    assert all(counts_y - how_many.sum(axis = 0) >= 0), 'Not enough instances from each class to compy with how_many'

    # Find indices for each class and shuffle them
    np.random.seed(random_state)
    wheres = {}
    for class_i in classes:
        w = np.where(y == class_i)[0]
        np.random.shuffle(w)
        wheres[class_i] = list(w)

    # Client -> list of (indices, columns)
    partition = [[] for _ in range(c_clients)]

    # For every class
    for i, class_i in enumerate(classes):

        # We distribute the corresponding indices to the clients
        prev = 0
        for client, ni in enumerate(how_many[:, i]):
            partition[client].extend(wheres[class_i][prev:prev+ni])
            added = len(wheres[class_i][prev:prev+ni])

            if debug:
                assert added == ni, f'added: {added} ni:{ni}'

            prev += ni 

    return [(np.array(ix), df.columns.to_list()) for ix in partition]


def dirichlet_partition(df, y, c_clients, alpha, only_with_labels = None, max_tries = 100, random_state = None):
    """
    Randomly partitions an array of labels y into a # c_clients of clients
    according to Dirichelet sampling with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    To guarantee that every 0 < alpha <= 1 can be met the total number of samples that can
    be sampled is set to the number of labels with the minimum frequency in y
    ('n_max_all_alphas' in the code). This may be too conservative but it's the
    easiest way to guarantee that samples_per_class_dirichlet doesn't over-assign a class
    (returning a matrix with a sum of column 0 that is greater than the # of instances of
    class 0, for example).

    alpha --> 0 implies very uneven sampling while alpha --> inf approaches uniform sampling.  

    Parameters
    ----------
    y : numpy array
        The numpy array of labels to be partitioned, assumed to be of integers 0 to
        # of classes -1.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the proportion of elements of y that the i-th client must sample.
        Therefore 

    only_with_labels : list(str) or None, optional
        If not None, only examples with these labels will be considered.
        
    Returns
    -------
    The partition as a dictionary: client id (int) -> array of indices (np.array).
    """
    n_classes = np.unique(y).size if only_with_labels is None else len(only_with_labels)
    tries, done = 0, False
    rs = random_state
    while tries < max_tries and not done:

        partition = _dirichlet_partition(
            df = df, y = y, c_clients = c_clients, alpha = alpha, only_with_labels = only_with_labels, random_state = rs
        )

        done = True
        for ix, _ in partition:
            v_counts = y.iloc[ix].value_counts()
            if v_counts.shape[0] < n_classes or v_counts.min() < MIN_PER_CLASS:
                done = False
                break
                
        rs = np.random.randint(0, 2**32 - 1)
        tries += 1
        # print(f'Try {tries} of {max_tries} with seed {rs}')

    assert done, 'Could not find a partition with enough instances per class'
    return partition


def _canonical_cols(df, skip_cols = []):

    out = []
    for c, t in df.dtypes.items():
        if t == bool and c.split('_')[0] not in skip_cols:
            out.append(c.split('_')[0])

        elif t != bool and c not in skip_cols:
            out.append(c)

    return sorted(list(set(out)))


def vertical_partitioning(df, target_col, c_clients, prop_cols, natural_col = None, random_state = None):

    # Canonical columns (i.e. ignore one-hot encoding)
    cols = _canonical_cols(df, skip_cols = [target_col] + ([natural_col] if natural_col is not None else []))
    
    np.random.seed(random_state)
    partition = []

    for _ in range(c_clients):
        canonical_chosen_cols = np.random.choice(cols, size =  int(prop_cols * len(cols)), replace = False)
        actual_cols = [c for c in df.columns if c.split('_')[0] in canonical_chosen_cols]
        actual_cols.append(target_col)  # Always include target
        partition.append((df.index, actual_cols))

    return partition