import pandas as pd

try:
    s = pd.Series(["a", "b"], dtype="string[pyarrow]")
    s[0] = 0
except Exception as e:
    print("pyarrow err:", type(e), str(e))

try:
    s2 = pd.Series(["a", "b"], dtype="string[python]")
    s2[0] = 0
except Exception as e:
    print("python str err:", type(e), str(e))

