import traceback
import importlib
try:
    m = importlib.import_module("app")
    print(getattr(m, "app"))
except Exception as e:
    traceback.print_exc()
