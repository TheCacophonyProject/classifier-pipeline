# bootstrap_keras.py
import os
import sys

# 1. Force the core TensorFlow engine to use legacy mode
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 2. Install tf_keras to ensure it is available
try:
    import tf_keras
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-keras tensorflow-model-optimization"])
    import tf_keras

# 3. Intercept and redirect the module path globally
sys.modules['tensorflow.keras'] = tf_keras
