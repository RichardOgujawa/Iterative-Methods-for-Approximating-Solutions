import numpy as np


def main():
    # Prompt the user to create the coefficient matrix, or use the default which is set manually below
    A, n, use_default_matrix = create_coefficient_matrix()
    # Based on what they chose to do above, they will either be prompted to create the constants vector or we will continue with the default constants vector
    b = create_constants_vector(n, use_default_matrix)
    # Get the rows from the matrix
    rows = A.shape[0]
    # Set the error margin constant
    E_MARG = 0.001
    # The error margin as a matrix
    E_MARG_MATRIX = np.array([E_MARG for _ in range(rows)])
    # Set the max number of iteraions we're willing to compute
    MAX_ITERATIONS = 10

    # Call the Jacobi method
    jacobi(A, b, E_MARG_MATRIX, MAX_ITERATIONS)
    # Seperator
    print()
    print("-" * 40)
    print()
    # Call the Gauss-Seidel method
    gauss_seidel(A, b, E_MARG, MAX_ITERATIONS)


# Iterator Functions (Jacobi and Gauss-Seidel)
def jacobi(A, b, E_MARG_MATRIX, MAX_ITERATIONS):
    print('Jacobi Method')
    # Get the number of rows in matrix A
    rows = A.shape[0]
    # Create the initial guess
    x0 = np.array([0. for _ in range(rows)])
    D, D_inv, L, L_inv, U = coefficient_matrix_conversions(A)
    # Add the upper and lower triangular matrices
    L_plus_U = L + U

    print('|| Initial Guess => ', x0)
    # Keep count of the number of iterations for the jacobi method with j_iteration_count
    j_iteration_count = 1
    # The stopping criteria for the jacobi method
    j_condition = True
    while j_condition:
        # Use the @ symbol for matrix multiplication
        x1 = D_inv @ (b - L_plus_U @ x0)

        print(f"|| Iteration {j_iteration_count} => {x1}")
        j_err_x = abs(x1 - x0)
        x0 = x1

        # Keep iterating as long as the error margins are greater than the acceptable error margin
        j_condition = np.all(j_err_x > E_MARG_MATRIX)

        # If we get to j_iteration_count gets as far as the max number of iterations
        if j_iteration_count == MAX_ITERATIONS:
            j_condition = False
            print(
                f"*** SOLUTIONS DIDN'T CONVERGE AFTER {MAX_ITERATIONS} ITERATIONS")
            return

        # Increment iteration counter
        j_iteration_count += 1

    print(f"*** CONVERGED IN {j_iteration_count - 1} ITERATIONS ***")


# Have the number of zeros in the initial guess matrix match the number of variables (rows in A)

# Iteration count to keep track of how many steps were needed to solve the system

def gauss_seidel(A, b, E_MARG, MAX_ITERATIONS):

    print('Gauss Seidel Method')
    rows = A.shape[0]
    gs_iteration_count = 1
    gs_condition = True

    # Initialise the solution vector x with zeros, its shape will be the same as vector b, and it will have a datatype double (used to store high precision decimal values)
    x = np.zeros_like(b, dtype=np.double)
    print('Initial Guess => ', x)

    # Initially induce a for loop whose condition is True, but will be changed to False once a condition is met in the loop
    while gs_condition:
        # Store the current value as the old one for future calculations.
        x_old = x.copy()

        # !!!HOW THE LOOP WORKS
        # We're updating each component of the solution vector independently.
        # The formula has a first summation term and second summation term. The first summation term, which in python looks like 'np.dot(A[i, :i], x[:i])'
        # The first summation term will multiply out the coefficients from A with the updated solutions from x
        # FIRST SUMMATION TERM -------------------------------
        #    A[i, :i] goes into the ith row and selects all of the coefficients before the diagonal, excluding the diagonal
        #    x[:i] selects all of the components that have been updated
        #    The dot product of the two gives us the contribution of the updated solution(s)
        # SECOND SUMMATION TERM ------------------------------
        #    A[i, [i+1]:] goes into the ith row and selects all of the components that haven't yet been updated
        #    x[(i+1):] selects the solutions that haven't yet been updated
        #    The dot product of the two gives us the contribution of the yet-to-be-updated solution(s)
        # We minus these two to see the change in the two estimates for the solution, at a certain point the change in this value will be so small that the entire vectors for the previous and updated solutions will be close enough to claim convergence, depending on our pre-defined error margin (E_MARG)
        # Finally we divide across by A[i, i] to isolate each variable
        for i in range(rows):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) -
                    np.dot(A[i, (i+1):], x_old[(i+1):])) / A[i, i]
        # Print the solution vector
        print(f"Iteration {gs_iteration_count} => {x}")

        # Stop condition
        gs_condition = np.linalg.norm(
            x - x_old, ord=np.inf) > E_MARG

        if gs_iteration_count == MAX_ITERATIONS:
            gs_condition = False
            print(
                f"*** SOLUTIONS DIDN'T CONVERGE AFTER {MAX_ITERATIONS} ITERATIONS")
            return
        # Increment iteration counter
        gs_iteration_count += 1

    print(f"*** CONVERGED IN {gs_iteration_count - 1} ITERATIONS ***")


# UTILITY FUNCTIONS --------------------------------
def create_coefficient_matrix():
    # This function allows the user to create their own coefficient matrix, or, if they so choose, use the default matrix below
    print("1. Create your square coefficient matrix (A), (or press 'Enter' to use the default matrix)")
    size_input = input(
        "What size (n) is your coefficient matrix (n x n)? ")
    use_default_matrix = False

    # If the user wants to just use the default matrix
    if size_input.strip() == '':
        A = np.array([[10,  -1, 3], [1,  11, -5], [2, -1, 13]])
        print("default coefficient matrix => ", A)
        n = A.shape[0]
        use_default_matrix = True
        return A, n, use_default_matrix

    # If the user inputs a matrix size that is too small, (i.e. less than 1) raise an exception
    if int(size_input) <= 1:
        raise ValueError("Please enter an n value greater than 1")

    # If the user enters a valid value, allow them to create a square coefficient matrix of that size
    n = int(size_input)
    A = np.zeros((n, n), dtype=np.double)
    rows, cols = A.shape

    for i in range(rows):
        for j in range(cols):
            A[i, j] = int(
                input("Which element do you want to put at position A{}? ".format([i, j])))

    print(f"*** YOUR SQUARE COEFFICIENT MATRIX IS: {A} ***")
    return A, n, use_default_matrix


def create_constants_vector(n, use_default_matrix):
    # This function allows the user to create their own constant matrix, or, if they previously had chosen to use the default matrix, then this function will by default use the default constants vector
    if use_default_matrix == False:
        print(f"2. Create your corresponding {n} x 1 constant vector (b)")
        b = np.zeros((n, 1), dtype=np.double)

        rows, cols = b.shape

        for i in range(rows):
            for j in range(cols):
                b[i, j] = int(
                    input("Which element do you want to put at position A{}? ".format([i, j])))
        # A 2d array is created, here we're flattening it to a 1d array before printing and returning it.
        print(f"*** YOUR CONSTANT VECTOR IS: {b.flatten()} ***")
        return b.flatten()

    # If the user wants to use the default matrix and b vector
    b = np.array([7, 5, 8])
    print("The default constant vector is =>", b)

    return b


def coefficient_matrix_conversions(A):
    # This functions turns the coefficients matrix, A, into the necessary components for the Jacobi formula (D, L, U, etc.)
    # np.diag is wrapped around np.diag to create a np.diag because np.diag(A) only returns the diagonal elements, if you want it in matrix form you have to call the method again passing in the diagonal elements as an argument.
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    # Get the lower triangular matrix and then fill in the diagonals with zeros
    # The order matters when it comes to getting the inverse of L, because filling in a lower triangular matrix with zero along the diagonals will result in singular matrix, i.e. a non-invertible matrix
    L = np.tril(A)
    L_inv = np.linalg.inv(L)
    np.fill_diagonal(L, 0)
    # Get the upper triangular matrix and then fill in the diagonals with zeros
    U = np.triu(A)
    np.fill_diagonal(U, 0)

    return D, D_inv, L, L_inv, U


if __name__ == "__main__":
    main()
