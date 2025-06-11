import numpy as np
import matplotlib.pyplot as plt


def step_jacobi_2D(u: np.ndarray, mask: np.ndarray, omega: float = 1.0) -> np.ndarray:
    u_new = u.copy()
    
    rows, cols = u.shape
    
    # Iterate over all pixels
    for i in range(1, rows-1):  
        for j in range(1, cols-1):
            if mask[i, j]:  # Update only masked pixels
                avg_neighbors = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                u_new[i, j] = (1 - omega) * u[i, j] + omega * avg_neighbors
    
    return u_new



def step_sor_2D(u: np.ndarray, mask: np.ndarray, omega: float = 1.0) -> np.ndarray:
    if u.shape != mask.shape:
        raise ValueError("u and mask must have the same shape")
    
    if np.any(mask):
        max_row, max_col = np.where(mask)[0].max(), np.where(mask)[1].max()
        rows, cols = u.shape
        if max_row >= rows or max_col >= cols:
            raise ValueError(f"Mask indices exceed image dimensions: {rows}x{cols}")
    
    u_new = u.copy()
    
    # Iterate over interior pixels
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if mask[i, j]:
                # Use new values for already-updated neighbors (i-1, j-1)
                avg_neighbors = 0.25 * (
                    u_new[i-1, j] + u[i+1, j] + u_new[i, j-1] + u[i, j+1]
                )
                # SOR update
                u_new[i, j] = (1 - omega) * u[i, j] + omega * avg_neighbors
    
    return u_new


def inpaint_jacobi(image: np.ndarray, mask: np.ndarray, tol: float = 1e-10, max_iter: int = 1000, omega: float = 1.0) -> tuple[np.ndarray, int]:
    u = image.copy()
    
    # Iterate until convergence or max_iter
    for it in range(max_iter):
        # Perform one Jacobi step
        u_new = step_jacobi_2D(u, mask, omega)
        
        delta = np.linalg.norm(u_new - u)
        u = u_new

        if delta < tol:
            return u, it + 1
    
    return u, max_iter

def inpaint_sor(image: np.ndarray, mask: np.ndarray, tol: float = 1e-10, max_iter: int = 1000, omega: float = 1.0) -> tuple[np.ndarray, int]:

    u = image.copy()
    
    # Iterate until convergence or max_iter
    for it in range(max_iter):
        u_old = u.copy()
        # Perform one SOR step
        u = step_sor_2D(u, mask, omega)
        
        delta = np.linalg.norm(u - u_old)
        
        # Check convergence
        if delta < tol:
            return u, it + 1
        
    return u, max_iter


def find_optimal_omega_sor(image: np.ndarray, mask: np.ndarray, omega_values: np.ndarray = None, nIter: int = 1000, abstol: float = 1e-10) -> tuple[np.ndarray, int]:

    if omega_values is None:
        omega_values = np.linspace(1.0, 1.95, 20)
    
    min_iters = nIter + 1
    omega_opt = omega_values[0]
    
    # Test each omega
    for omega in omega_values:
        _, iters = inpaint_sor(image, mask, tol=abstol, max_iter=nIter, omega=omega)
        # Update optimal omega if fewer iterations
        if iters < min_iters:
            min_iters = iters
            omega_opt = omega
    
    return omega_opt, min_iters


if __name__ == '__main__':
  # Load image
  image = np.loadtxt('image.txt')

  # Create mask
  mask = np.zeros_like(image, dtype=bool)
  mask[60:70, 30:70] = True
  
  # Create damaged image
  damaged = image.copy()
  damaged[mask] = 0.0

  # Refine with Jacobi
  refined_jacobi, it_jacobi = inpaint_jacobi(damaged, mask)

  # Refine with SOR
  refined_sor, it_sor = inpaint_sor(damaged, mask)

  # Plot
  fig, axs = plt.subplots(1, 4, figsize=(14, 4))
  axs[0].imshow(image, cmap='gray')
  axs[0].set_title("Original")
  axs[1].imshow(damaged, cmap='gray')
  axs[1].set_title("Damaged")
  axs[2].imshow(refined_jacobi, cmap='gray')
  axs[2].set_title("Jacobi")
  axs[3].imshow(refined_sor, cmap='gray')
  axs[3].set_title("SOR")
  for ax in axs:
    ax.axis('off')
  plt.tight_layout()
  plt.show()
