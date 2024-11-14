import os
import argparse
from ctranslate2.converters.transformers import TransformersConverter

# for faster-whisper
def convert_model(input_dir, output_dir):
    """Convert Whisper model to CTranslate2 format"""
    print(f"Converting model to CTranslate2 format")

    # Initialize the converter
    converter = TransformersConverter(
        model_name_or_path=input_dir,
    )

    # Convert the model
    converter.convert(
        output_dir=output_dir,
        force=True  
    )

    print(f"Model converted and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the Whisper model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the converted model")
    args = parser.parse_args()
    
    convert_model(args.input_dir, args.output_dir)
