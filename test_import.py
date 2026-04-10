import importlib
import traceback

try:
    module = importlib.import_module("app")
    app = getattr(module, "app")
    print("Success:", app)
except Exception as e:
    traceback.print_exc()
