import onnxruntime as ort
import numpy as np

class MockTensorRTEngine:
    """
    A mock TensorRT engine that wraps ONNXRuntime to simulate a TensorRT-like API facade
    for edge-compute deployment.
    """
    def __init__(self, onnx_model_path):
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[Mock TensorRT] Loaded engine from {onnx_model_path}")
        
    def execute(self, inputs):
        """
        Mock execute function
        inputs: numpy array
        """
        # Ensure inputs are float32
        inputs = np.array(inputs, dtype=np.float32)
        
        # Add batch dimension if necessary
        if len(inputs.shape) == 1:
            inputs = np.expand_dims(inputs, axis=0)
            
        outputs = self.session.run([self.output_name], {self.input_name: inputs})
        return outputs[0]

if __name__ == "__main__":
    import os
    if os.path.exists("deploy/pinn.onnx"):
        engine = MockTensorRTEngine("deploy/pinn.onnx")
        res = engine.execute(np.array([0.5, 0.1, 1.0]))
        print(f"PINN Inference result: {res}")
        
    if os.path.exists("deploy/rl_policy.onnx"):
        engine = MockTensorRTEngine("deploy/rl_policy.onnx")
        res = engine.execute(np.random.randn(67))
        print(f"RL Inference result: {res}")
