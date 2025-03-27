import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import constants  # Assuming constants.MODEL_ID is defined

def main():
    # Select device: use "mps" if available, else fall back to CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # When using MPS or CUDA, you can often use float16 for better performance,
    # but on CPU we stick with float32.
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    model_id = constants.MODEL_ID

    # Load model and move it to the selected device
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # When constructing the pipeline, note that the `device` argument is typically an
    # integer index (with -1 for CPU). Since we're using MPS, you should leave device as -1
    # and manually move the model to MPS as done above.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        # device=-1,  # Set to -1 since we've already moved the model to MPS
        return_timestamps=True
    )

    # Example usage:
    input_data = constants.INPUT_DATA
    outputs = pipe(input_data)
    print(outputs)

    # Writes to output/scene.txt
    with open("output/" + input_data, "w", encoding="utf-8") as f:
        f.write(outputs["text"])



if __name__ == "__main__":
    main()
