import numpy as np
import scipy.linalg


def lufact(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    Ak = A.copy()

    for k in range(n-1):
        U[k, :] = Ak[k, :]
        piv = U[k, k]
        if piv == 0:
            raise ZeroDivisionError(f"Pivot cero en k={k} (esta versión no pivotea).")
        L[:, k] = Ak[:, k] / piv
        Ak -= np.outer(L[:, k], U[k, :])

    U[n-1, n-1] = Ak[n-1, n-1]
    return L, U

def det_via_lu(A):
    """det(A) = prod(diag(U)) porque L es unipotente."""
    _, U = lufact(A)
    return float(np.prod(np.diag(U)))

# Opción estable con pivoteo
def det_via_plu(A):
    """Para matrices generales: det(A) = det(P) * prod(diag(U))."""
    P, L, U = scipy.linalg.lu(np.asarray(A, float), permute_l=False)
    detP = float(round(np.linalg.det(P)))  # ±1
    return detP * float(np.prod(np.diag(U)))

#  Verificación con tres matrices
A1 = np.array([[3, 1],
               [1, 2]], float)

A2 = np.array([[2, 1, 1],
               [4,-6, 0],
               [-2,7, 2]], float)

rng = np.random.default_rng(7)
A3 = rng.integers(-5, 6, size=(4,4)).astype(float)

for name, A in {"A1":A1, "A2":A2, "A3":A3}.items():
    det_np = float(np.linalg.det(A))
    try:
        det_lu  = det_via_lu(A)   # sin pivoteo
    except ZeroDivisionError as e:
        det_lu = str(e)
    det_plu = det_via_plu(A)      # con pivoteo (estable)

    print(f"{name}:")
    print(f"  det numpy   = {det_np:.12g}")
    print(f"  det via LU  = {det_lu}")
    print(f"  det via PLU = {det_plu:.12g}\n")

# Nota: 7.0
