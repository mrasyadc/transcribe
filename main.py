import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import constants

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = constants.MODEL_ID

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        # language="id",
        return_timestamps=True
    )

    result = pipe("input/input.mp3")
    # print(result["text"])

    # output the text
    with open("output/output.txt", "w", encoding="UTF-8") as f:
        f.write(result["text"])


if __name__ == "__main__":
    main()
