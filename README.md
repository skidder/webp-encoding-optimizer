# WebP Encoding Optimizer

A Python tool to find optimal WebP encoding settings for images. This tool analyzes various combinations of WebP encoding parameters to find the best balance between image quality, file size, and encoding time.

## Features

- Tests multiple WebP encoding configurations:
  - Different presets (default, picture, photo, drawing, icon, text)
  - Various quality levels
  - Multiple target dimensions
  - Both lossy and lossless compression
- Evaluates results using:
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
  - File size
  - Encoding time
- Provides detailed statistics for each configuration
- Supports batch processing of multiple images
- Option to preserve all output files for comparison
- Progress reporting during analysis
- Configurable weights for quality vs size vs speed

## Installation

1. Clone this repository:
```bash
git clone https://github.com/skidder/webp-encoding-optimizer.git
cd webp-encoding-optimizer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the WebP tools installed:
- On Ubuntu/Debian:
  ```bash
  sudo apt-get install webp
  ```
- On macOS:
  ```bash
  brew install webp
  ```
- On Windows:
  Download from [WebP Precompiled Utilities](https://developers.google.com/speed/webp/download)

## Usage

Basic usage:
```bash
python optimizer.py image.png
```

With custom options:
```bash
# Custom quality levels
python optimizer.py image.jpg --qualities 75 85 95

# Custom dimensions
python optimizer.py image.png --dimensions 44 44 88 88

# Process directory of images
python optimizer.py ./images --batch

# Preserve all output files
python optimizer.py image.png --preserve-outputs

# Custom weights for SSIM, encode time, and file size
python optimizer.py image.png --weights 0.4 0.2 0.4
```

### Available Options

- `--qualities`: Space-separated quality values (e.g., 75 85 95)
- `--presets`: Space-separated preset values (default, picture, photo, drawing, icon, text)
- `--dimensions`: Space-separated dimension values in pairs (e.g., 44 44 88 88)
- `--weights`: Three space-separated float values for SSIM, encode time, and file size weights
- `--batch`: Process all images in the input directory
- `--preserve-outputs`: Keep all generated WebP files in webp_results directory
- `--export-csv`: Export detailed statistics to a CSV file

### WebP Presets

- `default`: General-purpose preset
- `picture`: Digital picture, like portrait, inner shot
- `photo`: Outdoor photograph, with natural lighting
- `drawing`: Hand or line drawing, with high-contrast details
- `icon`: Small-sized colorful images
- `text`: Text-like images

## Output

The tool provides:
1. Optimal configuration for each target dimension
2. Quality metrics (SSIM and PSNR scores)
3. File size and encoding time
4. Statistical comparison of all tested configurations
5. Optional preserved output files in organized directories
6. Optional CSV export of all statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- WebP image format by Google
- Rich library for beautiful terminal output
- OpenCV and scikit-image for image analysis

## Author

Scott Kidder (@skidder)