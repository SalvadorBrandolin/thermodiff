from thermodiff import idx_function, k, l, n, nc, sum_components, sum_custom


def test_sum_components():
    phi_kl = idx_function(r"\phi")(k, l)

    sum1 = sum_components(phi_kl * n[k], k)
    sum2 = sum_custom(phi_kl * n[k], k, start=1, end=nc)

    assert sum1 == sum2
