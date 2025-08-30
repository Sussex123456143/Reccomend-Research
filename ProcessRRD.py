from flask import Flask, request, jsonify
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

@app.route('/run-notebook', methods=['POST'])
def run_notebook():
    user_input = request.json.get('input')
    # Load notebook
    with open('Recommend_Research_Database.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    # Inject user input (e.g., set a variable in the first cell)
    nb.cells[0].source = f"user_input = '{user_input}'\n" + nb.cells[0].source
    ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})
    # Extract output from last cell
    output = nb.cells[-1].outputs[0].text if nb.cells[-1].outputs else ""
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)