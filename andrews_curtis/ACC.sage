def ac_moves(pair, x, y):
    """
    Given a pair (u1, u2) (elements of G) and fixed generators x, y,
    return the set of all neighbors of (u1, u2) according to the AC-moves.
    """
    u1, u2 = pair
    moves = []

    moves.append((u1**(-1), u2))
    moves.append((u1, u2**(-1)))

    moves.append((u1 * u2, u2))
    moves.append((u1, u2 * u1))

    moves.append((u2 * u1, u2))
    moves.append((u1, u1 * u2))

    for g in [x, y, x**(-1), y**(-1)]:
        moves.append((g**(-1) * u1 * g, u2))
        moves.append((u1, g**(-1) * u2 * g))

    return moves


def ac_distance(x, y, AK_n=4, max_depth=20):
    """
    Given generators x and y (elements of a group G), compute the shortest 
    distance (i.e. minimal number of AC-moves) between the starting pair (x, y)
    and the target AK-pair (r_n(x, y), s_n(x, y)), where:

      r_n(x, y) = x*y*x*y^(-1)*x^(-1)*y^(-1)
      s_n(x, y) = x^n * y^(-n-1)

    max_depth: a safeguard parameter to limit the search depth.
    If no path is found within max_depth, returns None.
    """
    r = x * y * x * y**(-1) * x**(-1) * y**(-1)
    s = x**AK_n * y**(-AK_n-1)
    start = (x, y)
    target = (r, s)

    def elem_key(g):
        try:
            hash(g)
            return g
        except TypeError:
            # fallback for sage matrices (or any other unhashable):
            return tuple(g.list())

    def pair_key(pair):
        return (elem_key(pair[0]), elem_key(pair[1]))

    from collections import deque
    queue = deque([start])
    distance = {pair_key(start): 0}
    current_depth = 0

    while queue:
        current = queue.popleft()
        if current == target:
            return distance[curr_key]
        curr_key = pair_key(current)
        for neighbor in ac_moves(current, x, y):
            neigh_key = pair_key(neighbor)
            if neigh_key not in distance:
                distance[neigh_key] = distance[curr_key] + 1
                if distance[neigh_key] < max_depth:
                    queue.append(neighbor)
                    if distance[neigh_key] > current_depth:
                        current_depth = distance[neigh_key]
                        print("Current depth:", current_depth)
    return None


def ac_distance_PSL(n, q, AK_n=4, max_depth=20):
    """
    Computes the shortest AC-distance (number of AC-moves) from (x, y) to
    (r_{AK_n}(x, y), s_{AK_n}(x, y)) in PSL(n, q).
    """
    F = GF(q)

    S = SL(n, F)
    G = PSL(n, F)
    print("Group PSL(%s, %s): order = %s" % (n, q, G.order()))

    # This approach maps the generators of SL(n, q) to those of PSL(n, q)
    f = S.Hom(G)(G.gens())
    assert [f(g) for g in S.gens()] == list(G.gens())

    if n == 2:
        X = matrix(F, [[1, 1], [0, 1]])
        Y = matrix(F, [[1, 0], [1, 1]])
    elif n == 3:
        X = matrix(F, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        Y = matrix(F, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    else:
        print("No generators defined")
        return

    x = f(S(X))
    y = f(S(Y))

    dist = ac_distance(x, y, AK_n=AK_n, max_depth=max_depth)
    if dist is None:
        print("No path found within depth", max_depth)
    else:
        print("Shortest distance (number of AC-moves):", dist)


def ac_distance_F4(q, AK_n=4, max_depth=20):
    """
    Computes the shortest AC-distance (number of AC-moves) from (x, y) to
    (r_{AK_n}(x, y), s_{AK_n}(x, y)) in F_4(q).
    """
    F = GF(q)

    print("Group F_4(%s)" % q)

    # matrix of transformation from simple roots basis to standard R^4 basis
    C = matrix([[0, 0, 0, 1/2],
                [1, 0, 0, -1/2],
                [-1, 1, 0, -1/2],
                [0, -1, 1, -1/2]])

    # positive roots (first 4 are the simple roots)
    # in simple roots basis
    roots = [vector([1, 0, 0, 0]), vector([0, 1, 0, 0]), vector([0, 0, 1, 0]), vector([0, 0, 0, 1]), vector([1, 1, 0, 0]), vector([0, 1, 1, 0]),
             vector([0, 0, 1, 1]), vector([1, 1, 1, 0]), vector([0, 1, 1, 1]), vector([0, 1, 2, 0]), vector([0, 1, 2, 1]), vector([1, 1, 1, 1]),
             vector([1, 1, 2, 0]), vector([0, 1, 2, 2]), vector([1, 1, 2, 1]), vector([1, 2, 2, 0]), vector([1, 1, 2, 2]), vector([1, 2, 2, 1]),
             vector([1, 2, 2, 2]), vector([1, 2, 3, 1]), vector([1, 2, 3, 2]), vector([1, 2, 4, 2]), vector([1, 3, 4, 2]), vector([2, 3, 4, 2])]
    # transforming to standard basis
    for i in range(24):
        roots[i] = C * roots[i]
    # negative roots
    for i in range(24):
        roots.append(-roots[i])

    dual_simple_roots = []
    for i in range(4):
        a = roots[i]
        dual_simple_roots.append((2 / (a * a)) * a)
    D = matrix(dual_simple_roots).transpose()

    def is_root(root):
        return root in roots

    def root_sign(root):
        if roots.index(root) < 24:
            return 1
        return -1

    NN = [[0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
          [-1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0],
          [0, -1, 0, 1, -1, -2, 0, -2, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, 0, -2, 0, 0, 0],
          [0, 0, -1, 0, 0, -1, 0, -1, 0, -1, -2, 0, -1, 0, -2, -1, 0, -2, 0, -1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [-1, 0, 2, 1, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 2, 0, 0, 0],
          [0, -1, 0, 0, -1, -1, 0, -1, -2, 0, 0, -2, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0],
          [0, 0, 2, 1, 0, -2, 1, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0],
          [-1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, -2, 0, 0, 0, 0],
          [-1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0],
          [-1, 0, 0, 2, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, -1, 2, 0, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
          [0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [-1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 2, 0, 1, 0, 0, -1, 0, -2, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 2, 0, 0, -1, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, -2, 0, 2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 2, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # N_ab = -N_ba = -N_-a,-b
    # a + b + c = 0 ==> N_ab / (c,c) = N_bc / (a,a) = N_ca / (b,b)
    # a + b + c + d = 0 and no pair is antipodal ==> N_ab*N_cd / (a+b,a+b) + N_bc*N_ad / (b+c,b+c) + N_ca*N_bd / (c+a,c+a) = 0
    N = [[0] * 48 for _ in range(48)]
    for i in range(48):
        for j in range(48):
            if not is_root(roots[i] + roots[j]):
                continue
            r = 0
            while is_root(roots[j] - r * roots[i]):
                r += 1
            N[i][j] = r

    for i in range(24):
        for j in range(24):
            N[i][j] = NN[i][j]

    # now signs are set for all pairs of roots a, b where 0 < a < b  (special pairs)
    # setting signs for all pairs:
    for i in range(48):
        for j in range(48):
            if not is_root(roots[i] + roots[j]):
                continue
            sg, ii, jj = 1, i, j
            if jj >= 24:
                ii, jj = roots.index(-roots[ii]), roots.index(-roots[jj])
                sg *= -1
            if ii >= 24:
                if root_sign(roots[ii] + roots[jj]) > 0:
                    ii, jj = roots.index(-roots[ii]), roots.index(roots[ii] + roots[jj])
                else:
                    ii, jj = jj, roots.index(-(roots[ii] + roots[jj]))
            # here roots[ii], roots[jj] > 0
            if ii > jj:
                ii, jj = jj, ii
                sg *= -1
            # roots[ii], roots[jj] is now a special pair (for which the sign of N is already set)
            N[i][j] = abs(N[i][j]) * sg * sgn(N[ii][jj])

    def cartan(root1, root2):
        return 2 * (root1 * root2) / (root2 * root2)

    def ad(i, j):
        if i >= 48 and j >= 48:
            # ad[h_i, h_j] = 0
            return vector([0] * 52)
        if i >= 48 and j < 48:
            # ad[h_i, x_j] = <a_j, a_i> * x_j
            v = [0] * 52
            v[j] = cartan(roots[j], roots[i - 48])
            return vector(v)
        if i < 48 and j >= 48:
            # ad[x_i, h_j]
            return -ad(j, i)

        # i, j < 48
        if i == j:
            return vector([0] * 52)
        if roots[i] == -roots[j]:
            # ad[x_a, x_-a] = h_a
            a = roots[i]
            aa = (2 / (a * a)) * a  # dual root
            bb = D**(-1) * aa
            return vector([0] * 48 + list(bb))
        if is_root(roots[i] + roots[j]):
            v = [0] * 52
            v[roots.index(roots[i] + roots[j])] = N[i][j]
            return vector(v)
        else:
            return vector([0] * 52)

    def ad_x(i):
        l = []
        for j in range(52):
            l.append(ad(i, j))
        return matrix(l).transpose()

    X = []
    for i in range(52):
        X.append(ad_x(i))

    def ring_commute(a, b):
        return a * b - b * a

    def check_basis_relations():
        for i in range(48):
            for j in range(48):
                if is_root(roots[i] + roots[j]):
                    assert ring_commute(X[i], X[j]) == N[i][j] * X[roots.index(roots[i] + roots[j])]
                elif roots[i] == -roots[j]:
                    a = roots[i]
                    aa = (2 / (a * a)) * a  # dual root
                    bb = D**(-1) * aa
                    assert ring_commute(X[i], X[j]) == bb[0] * X[48] + bb[1] * X[49] + bb[2] * X[50] + bb[3] * X[51]
                else:
                    assert ring_commute(X[i], X[j]) == 0
        for i in range(48, 52):
            for j in range(48, 52):
                assert ring_commute(X[i], X[j]) == 0
        for i in range(4):
            for j in range(48):
                assert ring_commute(X[48 + i], X[j]) == cartan(roots[j], roots[i]) * X[j]

    check_basis_relations()

    def x(i, t=1):
        t_adx = t * ad_x(i)
        #return t_adx.exp()
        A = (t_adx)**0
        M = A * 0
        k, fact_k = 0, 1
        while A != 0:
            M += A / fact_k
            A *= t_adx
            k += 1
            fact_k *= k
        return M

    x1 = x(0).change_ring(F)
    x2 = x(1).change_ring(F)

    dist = ac_distance(x1, x2, AK_n=AK_n, max_depth=max_depth)
    if dist is None:
        print("No path found within depth", max_depth)
    else:
        print("Shortest distance (number of AC-moves):", dist)
