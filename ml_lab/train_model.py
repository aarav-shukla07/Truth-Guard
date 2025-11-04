import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import os

# --- 1. Our Sample Dataset ---
# In a real project, we'd have thousands of these.
# Label 0 = Good Code, Label 1 = Bad Code (Bloated)
code_data = [
    ("def add(a, b):\n  return a + b", 0),
    ("def get_user(id):\n  user = db.find(id)\n  return user", 0),
    ("def main():\n  print('hello')", 0),
    
    ("def process_data(a, b, c, d, e, f):\n  print(a)\n  if a > 10:\n    print(b)\n    if b > 20:\n      print(c)\n      if c > 30:\n        print(d)\n        if d > 40:\n          print(e)\n          if e > 50:\n            print(f)\n            # ... many more lines\n            return a + b + c + d + e + f", 1),
    
    ("def huge_function(arg1, arg2, arg3, arg4):\n  var1 = arg1 * 2\n  var2 = arg2 * 3\n  var3 = arg3 * 4\n  var4 = arg4 * 5\n  var5 = 'temp'\n  var6 = 'another'\n  var7 = [1,2,3,4,5]\n  # ...imagine 50 more lines of logic\n  if var1 > var2:\n    print('hello')\n  elif var2 > var3:\n    print('world')\n  else:\n    print('done')\n  return var1 + var2 + var3 + var4", 1),

    ("def another_bad_one(user, data, config, settings, extra, more, stuff):\n  # This function just has way too many parameters.\n  # It indicates poor design (code smell).\n  line1 = user + data\n  line2 = config + settings\n  line3 = extra + more\n  if stuff:\n    return line1 + line2 + line3", 1)
]

# --- 2. Feature Engineering ---
# This is our function to convert raw code into numbers.
def extract_features(code):
    # Feature 1: Line Count
    line_count = len(code.split('\n'))
    
    # Feature 2: Parameter Count
    # A simple regex to find parameters in 'def name(p1, p2):'
    match = re.search(r'def \w+\((.*?)\):', code)
    param_count = 0
    if match:
        params = match.group(1).strip()
        if params: # Check if it's not empty
            param_count = len(params.split(','))
            
    # Feature 3: Complexity (count of if/for/while)
    complexity = len(re.findall(r'(if |for |while )', code))
    
    # Feature 4: Variable Declarations (simple check for '=')
    var_count = len(re.findall(r'\w+ = ', code))

    return [line_count, param_count, complexity, var_count]

# Process the dataset
features = []
labels = []
for code, label in code_data:
    features.append(extract_features(code))
    labels.append(label)

# Convert to NumPy arrays for scikit-learn
X = np.array(features, dtype=np.float32)
y = np.array(labels)

print("--- Feature Data (X) ---")
print(X)
print("\n--- Labels (y) ---")
print(y)

# --- 3. Model Training ---
print("\n--- Training Model ---")
# We split our tiny dataset just to show the process
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# We use a RandomForest. It's powerful, small, and fast.
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy on Test Set: {model.score(X_test, y_test) * 100:.2f}%")

# --- 4. Model Export to ONNX ---
print("\n--- Exporting to ONNX ---")
model_file_name = "bug_detector.onnx"
feature_count = X.shape[1] # This is 4

# This tells ONNX what kind of data to expect:
# [None, feature_count] means [any_batch_size, 4_features]
initial_type = [('float_input', FloatTensorType([None, feature_count]))]

# Convert the model
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the model to a file
with open(model_file_name, "wb") as f:
    f.write(onnx_model.SerializeToString())
    
print(f"Successfully exported model to: {model_file_name}")
print(f"File size: {os.path.getsize(model_file_name) / 1024:.2f} KB")


# --- 5. Verification (CRITICAL STEP) ---
# Let's test the .onnx file we just made to prove it works.

print("\n--- Verifying ONNX Model ---")

# Create a sample "bad" function to test
test_code_snippet = """
def very_bad_function(p1, p2, p3, p4, p5, p6, p7):
  a = 1
  b = 2
  c = 3
  if a > b:
    if b > c:
      if c > a:
        print('too deep')
  for i in range(10):
    print(i)
  return 'this is a bloated function'
"""

# 1. Extract features just like we will in VS Code
test_features = np.array([extract_features(test_code_snippet)], dtype=np.float32)
print(f"Test Features: {test_features}")

# 2. Load the .onnx model with onnxruntime
sess = rt.InferenceSession(model_file_name)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# 3. Get a prediction
# The result is [prediction_label, probabilities]
# We only care about the prediction (index 0)
prediction = sess.run([label_name], {input_name: test_features})[0]

print(f"\nPrediction for test code: {prediction[0]}")
if prediction[0] == 1:
    print("VERIFICATION SUCCESS: The ONNX model correctly classified the code as 'Bad'.")
else:
    print("VERIFICATION FAILED: The ONNX model classified the code as 'Good'.")