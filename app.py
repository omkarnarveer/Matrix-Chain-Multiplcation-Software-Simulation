from flask import Flask, render_template, request
import concurrent.futures

app = Flask(__name__)

def matrix_chain_order(p):
    """
    Matrix Chain Order with Multi-threading for performance.
    p: List of matrix dimensions
    Returns the minimum number of scalar multiplications, the parenthesization order, the number of subproblems,
    and matrix multiplication addresses (steps).
    """
    n = len(p) - 1  # Number of matrices
    dp = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]

    # Multi-threaded dynamic programming approach
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        # First, calculate for the smaller chains
        for l in range(2, n + 1):  # l is the chain length
            futures.append(executor.submit(process_chain, l, n, p, dp, s))
        
        # Wait for all futures to finish
        concurrent.futures.wait(futures)

    # Calculate the number of subproblems
    num_subproblems = (n * (n - 1)) // 2  # Number of subproblems is n(n-1)/2

    # Generate matrix names
    matrix_names = [f"M{i+1}" for i in range(n)]

    # Matrix multiplication steps (addresses)
    multiplication_steps = generate_multiplication_steps(s, matrix_names, 0, n - 1)

    return dp[0][n - 1], s, num_subproblems, matrix_names, multiplication_steps

def process_chain(l, n, p, dp, s):
    """
    Helper function to compute dp[i][j] using multi-threading.
    """
    for i in range(n - l + 1):
        j = i + l - 1
        dp[i][j] = float('inf')
        for k in range(i, j):
            q = dp[i][k] + dp[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
            if q < dp[i][j]:
                dp[i][j] = q
                s[i][j] = k

def generate_multiplication_steps(s, matrix_names, i, j):
    """
    Recursively generate the multiplication steps (addresses) for the optimal order.
    """
    if i == j:
        return [matrix_names[i]]
    else:
        k = s[i][j]
        left_order = generate_multiplication_steps(s, matrix_names, i, k)
        right_order = generate_multiplication_steps(s, matrix_names, k + 1, j)
        return left_order + right_order

def get_optimal_order(s, i, j):
    """
    Helper function to construct the optimal parenthesization order
    from the split table s.
    """
    if i == j:
        return f"M{i+1}"
    else:
        return f"({get_optimal_order(s, i, s[i][j])} x {get_optimal_order(s, s[i][j] + 1, j)})"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_dims = None
    num_subproblems = None
    matrix_names = None
    multiplication_steps = None
    if request.method == "POST":
        try:
            # Get the matrix dimensions from the form (as comma-separated values)
            input_dims = request.form['dimensions']
            dimensions = list(map(int, input_dims.split(',')))

            # Ensure that we have the correct number of dimensions
            if len(dimensions) < 2:
                raise ValueError("Please provide at least two matrices.")

            # Compute the minimum number of scalar multiplications, subproblems, matrix names, and multiplication steps
            min_cost, split_table, num_subproblems, matrix_names, multiplication_steps = matrix_chain_order(dimensions)

            # Create the optimal parenthesization string
            optimal_order = get_optimal_order(split_table, 0, len(dimensions) - 2)

            result = {
                'min_cost': min_cost,
                'optimal_order': optimal_order,
                'num_subproblems': num_subproblems,
                'matrix_names': matrix_names,
                'multiplication_steps': multiplication_steps
            }

        except ValueError as e:
            result = str(e)

    return render_template("index.html", result=result, input_dims=input_dims)

if __name__ == "__main__":
    app.run(debug=True)
