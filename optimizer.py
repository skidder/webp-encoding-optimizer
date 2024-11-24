import subprocess
import os
import time
import threading
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
import json
import argparse

class WebPOptimizer:
    def __init__(self, input_image_path, config=None):
        """Initialize the optimizer with an input image path."""
        self.input_image = input_image_path
        
        # Define default configuration
        default_config = {
            'presets': ['default', 'picture', 'photo', 'drawing', 'icon', 'text'],
            'qualities': [50, 75, 85, 90, 95],
            'target_dimensions': [(44, 44), (88, 88), (100, 100), (400, 400)],
            'weights': {
                'ssim': 0.4,
                'encode_time': 0.2,
                'file_size': 0.4
            },
            'preserve_outputs': False
        }
        
        # Merge provided config with defaults
        if config:
            self.config = default_config.copy()
            self.config.update(config)
        else:
            self.config = default_config
        
        self.results = []
        self._results_lock = threading.Lock()
        
        # Create temp directory
        self.temp_dir = "temp_webp_optimizer"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        self.cache_dir = ".webp_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def resize_image(self, target_size):
        """
        Resize input image to target dimensions while maintaining aspect ratio.
        Returns None if resizing fails.
        """
        temp_path = os.path.join(self.temp_dir, f"temp_{target_size[0]}x{target_size[1]}.webp")
        try:
            with Image.open(self.input_image) as img:
                print(f"Input image mode: {img.mode}, format: {img.format}, size: {img.size}")
                
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGB')
                
                # Calculate aspect ratio
                aspect_ratio = img.size[0] / img.size[1]
                target_aspect = target_size[0] / target_size[1]
                
                if aspect_ratio > target_aspect:
                    new_width = target_size[0]
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = target_size[1]
                    new_width = int(new_height * aspect_ratio)
                
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                final_img = Image.new('RGB', target_size, (0, 0, 0))
                paste_x = (target_size[0] - new_width) // 2
                paste_y = (target_size[1] - new_height) // 2
                final_img.paste(resized, (paste_x, paste_y))
                
                # Save directly as WebP - either use lossless or quality, not both
                final_img.save(temp_path, 'WEBP', lossless=True)
                os.fsync(os.open(temp_path, os.O_RDONLY))
                
                # Verify file exists and is readable
                if not os.path.exists(temp_path):
                    raise IOError(f"Failed to create temporary file: {temp_path}")
                
                with open(temp_path, 'rb') as f:
                    if len(f.read(1)) == 0:
                        raise IOError(f"Temporary file is empty: {temp_path}")
                
                return temp_path
                
        except Exception as e:
            print(f"Error resizing image to {target_size}:")
            print(f"Error: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return None

    def encode_webp(self, input_path, preset, quality, output_path):
        """Encode an image to WebP using specified parameters."""
        start_time = time.time()
        
        # Create results directory if preserving outputs
        if self.config.get('preserve_outputs'):
            results_dir = "webp_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Get input image name without extension
            input_name = os.path.splitext(os.path.basename(self.input_image))[0]
            # Get dimensions from the output path
            dim_match = os.path.basename(output_path).split('_')[1]  # Gets "44x44" from the filename
            
            # Create organized paths
            output_base = os.path.join(results_dir, input_name, dim_match)
            if not os.path.exists(output_base):
                os.makedirs(output_base)
            
            # Create descriptive filenames
            lossy_name = f"{preset}_q{quality}.webp"
            lossless_name = f"{preset}_lossless.webp"
            
            # Update output paths
            lossy_path = os.path.join(output_base, lossy_name)
            lossless_path = os.path.join(output_base, lossless_name)
            
            cmds = [
                ['cwebp', '-preset', preset, '-q', str(quality), input_path, '-o', lossy_path],
                ['cwebp', '-preset', preset, '-lossless', input_path, '-o', lossless_path]
            ]
        else:
            cmds = [
                ['cwebp', '-preset', preset, '-q', str(quality), input_path, '-o', output_path],
                ['cwebp', '-preset', preset, '-lossless', input_path, '-o', f"{output_path}_lossless"]
            ]
        
        results = []
        for cmd in cmds:
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                encode_time = time.time() - start_time
                output_file = cmd[-1]
                file_size = os.path.getsize(output_file) / 1024  # Size in KB
                
                # Calculate quality metrics for this version
                quality_metrics = self.calculate_image_quality(input_path, output_file)
                
                results.append({
                    'success': True,
                    'encode_time': encode_time,
                    'file_size': file_size,
                    'is_lossless': '-lossless' in cmd,
                    'output_path': output_file,
                    'ssim': quality_metrics['ssim'],
                    'psnr': quality_metrics['psnr']
                })
            except Exception as e:
                print(f"Error with command {' '.join(cmd)}: {str(e)}")
        
        if not results:
            return {
                'success': False,
                'encode_time': 0,
                'file_size': 0
            }
        
        # Score each version based on quality and size
        for result in results:
            result['score'] = (
                result['ssim'] * self.config['weights']['ssim'] +
                (1 - (result['file_size'] / max(r['file_size'] for r in results))) * self.config['weights']['file_size'] +
                (1 - (result['encode_time'] / max(r['encode_time'] for r in results))) * self.config['weights']['encode_time']
            )
        
        # Select the best version
        best_result = max(results, key=lambda x: x['score'])
        
        # Modified cleanup section
        if not self.config.get('preserve_outputs'):
            # Clean up the unused version
            for result in results:
                if result['output_path'] != best_result['output_path']:
                    try:
                        os.remove(result['output_path'])
                    except Exception as e:
                        print(f"Error removing unused version: {str(e)}")
        
        return {
            'success': True,
            'encode_time': best_result['encode_time'],
            'file_size': best_result['file_size'],
            'is_lossless': best_result['is_lossless'],
            'ssim': best_result['ssim'],
            'psnr': best_result['psnr'],
            'output_path': best_result['output_path'] if self.config.get('preserve_outputs') else None
        }

    def calculate_image_quality(self, original_path, encoded_path):
        """Calculate quality metrics between original and encoded images."""
        try:
            # Read images
            original = cv2.imread(original_path)
            encoded = cv2.imread(encoded_path)
            
            if original is None or encoded is None:
                print(f"Error reading images for quality calculation")
                return {'ssim': 0, 'psnr': 0}
            
            # Ensure same size for comparison
            if original.shape != encoded.shape:
                encoded = cv2.resize(encoded, (original.shape[1], original.shape[0]))
            
            # Calculate SSIM with smaller window size for small images
            min_dim = min(original.shape[0], original.shape[1])
            win_size = min(3, min_dim - 1 if min_dim % 2 == 0 else min_dim)
            if win_size < 3:
                # Image too small for SSIM, use MSE only
                mse = np.mean((original - encoded) ** 2)
                return {
                    'ssim': 1.0 - (mse / (255.0 ** 2)),  # Normalize MSE to 0-1 range
                    'psnr': float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
                }
            
            # Calculate SSIM (1.0 is perfect similarity)
            ssim_score = ssim(
                original, 
                encoded, 
                multichannel=True,
                win_size=win_size,
                channel_axis=2  # Specify channel axis for RGB images
            )
            
            # Calculate PSNR (higher is better)
            mse = np.mean((original - encoded) ** 2)
            psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
            
            return {
                'ssim': ssim_score,
                'psnr': psnr
            }
        except Exception as e:
            print(f"Error calculating image quality: {str(e)}")
            return {'ssim': 0, 'psnr': 0}

    def analyze_single_configuration(self, config):
        """Analyze a single encoding configuration."""
        dimension, preset, quality = config
        
        # Create unique filenames for this specific configuration
        config_id = f"{dimension[0]}x{dimension[1]}_{preset}_{quality}"
        temp_input = os.path.join(self.temp_dir, f"temp_input_{config_id}.webp")
        output_path = os.path.join(self.temp_dir, f"output_{config_id}.webp")
        
        try:
            # Resize image with unique temp file
            with Image.open(self.input_image) as img:
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGB')
                
                # Calculate aspect ratio and resize
                aspect_ratio = img.size[0] / img.size[1]
                target_aspect = dimension[0] / dimension[1]
                
                if aspect_ratio > target_aspect:
                    new_width = dimension[0]
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = dimension[1]
                    new_width = int(new_height * aspect_ratio)
                
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                final_img = Image.new('RGB', dimension, (0, 0, 0))
                paste_x = (dimension[0] - new_width) // 2
                paste_y = (dimension[1] - new_height) // 2
                final_img.paste(resized, (paste_x, paste_y))
                
                # Save directly to temp file
                temp_png = os.path.join(self.temp_dir, f"temp_png_{config_id}.png")
                final_img.save(temp_png, 'PNG')
                
                # Convert to WebP using cwebp
                result = self.encode_webp(temp_png, preset, quality, output_path)
                
                if result['success']:
                    # Use thread-safe list append
                    with self._results_lock:
                        self.results.append({
                            'width': dimension[0],
                            'height': dimension[1],
                            'preset': preset,
                            'quality': quality,
                            'encode_time': result['encode_time'],
                            'file_size': result['file_size'],
                            'ssim': result['ssim'],
                            'psnr': result['psnr'],
                            'is_lossless': result['is_lossless'],
                            'output_path': result.get('output_path')
                        })
                
                # Clean up temporary files immediately
                for path in [temp_png, output_path]:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception as e:
                        print(f"Error removing temporary file {path}: {str(e)}")
                        
        except Exception as e:
            print(f"Error processing configuration {config_id}: {str(e)}")
            for path in [temp_input, output_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Error removing temporary file {path}: {str(e)}")

    def run_analysis(self):
        """Run the complete analysis across all configurations."""
        configurations = [
            (dim, preset, quality)
            for dim in self.config['target_dimensions']
            for preset in self.config['presets']
            for quality in self.config['qualities']
        ]
        
        total = len(configurations)
        console = Console()
        
        with console.status("[bold green]Analyzing configurations...") as status:
            with ThreadPoolExecutor(max_workers=4) as executor:
                for i, _ in enumerate(executor.map(self.analyze_single_configuration, configurations)):
                    progress = (i + 1) / total * 100
                    status.update(f"[bold green]Progress: {progress:.1f}% ({i+1}/{total} configurations)")
                    
                    # Optional: Show current configuration being processed
                    if i < total - 1:
                        next_config = configurations[i + 1]
                        status.console.print(
                            f"Next: {next_config[0]}px, preset: {next_config[1]}, quality: {next_config[2]}", 
                            style="dim"
                        )
        
        console.print("\n[bold green]✓[/bold green] Analysis complete!")
        return self.get_analysis_results()

    def get_analysis_results(self):
        """Process and return analysis results."""
        df = pd.DataFrame(self.results)
        
        if df.empty:
            print("No successful encoding results were found.")
            return []
        
        # Calculate efficiency scores
        df['efficiency_score'] = (
            (1 - df['ssim']) * self.config['weights']['ssim'] +
            (df['encode_time'] / df['encode_time'].max()) * self.config['weights']['encode_time'] +
            (df['file_size'] / (df['width'] * df['height']) / 
             (df['file_size'] / (df['width'] * df['height'])).max()) * self.config['weights']['file_size']
        )
        
        # Find optimal configurations for each dimension
        df['is_optimal'] = False
        for dimension in self.config['target_dimensions']:
            mask = (df['width'] == dimension[0]) & (df['height'] == dimension[1])
            if mask.any():
                optimal_idx = df[mask]['efficiency_score'].idxmin()
                df.loc[optimal_idx, 'is_optimal'] = True
        
        # Export raw results to CSV if requested
        if self.config.get('export_csv'):
            input_name = os.path.splitext(os.path.basename(self.input_image))[0]
            csv_filename = f"webp_results/{input_name}_statistics.csv"
            os.makedirs('webp_results', exist_ok=True)
            
            # Add relative metrics
            df['size_per_pixel'] = df['file_size'] / (df['width'] * df['height'])
            df['relative_size'] = df['file_size'] / df['file_size'].max()
            df['relative_time'] = df['encode_time'] / df['encode_time'].max()
            
            # Add rank within dimension group
            df['dimension_rank'] = df.groupby(['width', 'height'])['efficiency_score'].rank()
            
            # Reorder columns for better readability
            columns = [
                # Identification
                'width', 'height', 'preset', 'quality', 'is_lossless',
                # Primary metrics
                'file_size', 'encode_time', 'ssim', 'psnr',
                # Derived metrics
                'size_per_pixel', 'relative_size', 'relative_time',
                # Analysis results
                'efficiency_score', 'dimension_rank', 'is_optimal',
                # File info
                'output_path'
            ]
            
            df[columns].to_csv(csv_filename, index=False)
            print(f"\nStatistics exported to: {csv_filename}")
        
        # Get best configurations for each dimension
        best_configs = []
        for dimension in self.config['target_dimensions']:
            dimension_df = df[
                (df['width'] == dimension[0]) & 
                (df['height'] == dimension[1])
            ]
            if dimension_df.empty:
                continue
            
            best_row = dimension_df.loc[dimension_df['efficiency_score'].idxmin()]
            
            # Calculate stats for this dimension
            stats = {
                'ssim': {
                    'min': round(dimension_df['ssim'].min(), 3),
                    'max': round(dimension_df['ssim'].max(), 3),
                    'avg': round(dimension_df['ssim'].mean(), 3)
                },
                'encode_time': {
                    'min': round(dimension_df['encode_time'].min(), 3),
                    'max': round(dimension_df['encode_time'].max(), 3),
                    'avg': round(dimension_df['encode_time'].mean(), 3)
                },
                'file_size': {
                    'min': round(dimension_df['file_size'].min(), 2),
                    'max': round(dimension_df['file_size'].max(), 2),
                    'avg': round(dimension_df['file_size'].mean(), 2)
                }
            }
            
            best_configs.append({
                'dimensions': f"{dimension[0]}x{dimension[1]}",
                'best_preset': best_row['preset'],
                'best_quality': best_row['quality'],
                'encode_time': round(best_row['encode_time'], 3),
                'file_size_kb': round(best_row['file_size'], 2),
                'ssim': round(best_row['ssim'], 3),
                'psnr': round(best_row['psnr'], 3),
                'is_lossless': best_row['is_lossless'],
                'output_path': best_row['output_path'],
                'stats': stats
            })
            
        return best_configs

    def __del__(self):
        """Cleanup temporary directory when object is destroyed."""
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
                os.rmdir(self.temp_dir)
        except:
            pass

    def set_dimensions(self, dimensions):
        self.config['target_dimensions'] = dimensions
        return self
        
    def set_qualities(self, qualities):
        self.config['qualities'] = qualities
        return self
        
    def set_presets(self, presets):
        self.config['presets'] = presets
        return self

    def get_cached_result(self, config_id):
        cache_path = os.path.join(self.cache_dir, f"{config_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

class WebPBatchOptimizer:
    def __init__(self, input_directory):
        self.input_dir = input_directory
        self.optimizer = None
        
    def process_directory(self):
        results = {}
        for file in os.listdir(self.input_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(self.input_dir, file)
                self.optimizer = WebPOptimizer(file_path)
                results[file] = self.optimizer.run_analysis()
        return results

def main():
    """Main function to demonstrate usage."""
    import argparse
    import sys
    console = Console()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="WebP Image Optimizer - Find optimal encoding settings for WebP images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimizer.py image.png                     # Basic usage
  python optimizer.py image.jpg --qualities 75 85   # Custom quality levels
  python optimizer.py image.png --batch             # Process all images in directory
  python optimizer.py image.png --dimensions 44 88  # Custom dimensions
  
Presets:
  - default: General-purpose preset
  - picture: Digital picture, like portrait, inner shot
  - photo:   Outdoor photograph, with natural lighting
  - drawing: Hand or line drawing, with high-contrast details
  - icon:    Small-sized colorful images
  - text:    Text-like images
        """
    )

    # Add arguments
    parser.add_argument('input', help='Input image path or directory (if --batch is used)')
    parser.add_argument('--batch', action='store_true', help='Process all images in directory')
    parser.add_argument('--qualities', nargs='+', type=int, 
                       help='Space-separated quality values (e.g., 75 85 95)')
    parser.add_argument('--presets', nargs='+', choices=['default', 'picture', 'photo', 'drawing', 'icon', 'text'],
                       help='Space-separated preset values')
    parser.add_argument('--dimensions', nargs='+', type=int,
                       help='Space-separated dimension values (e.g., 44 88)')
    parser.add_argument('--weights', nargs=3, type=float, metavar=('SSIM', 'TIME', 'SIZE'),
                       help='Weights for SSIM, encode time, and file size (e.g., 0.4 0.2 0.4)')
    parser.add_argument('--preserve-outputs', action='store_true',
                       help='Preserve all output files in webp_results directory')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export all statistics to a CSV file in webp_results directory')
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input path '{args.input}' does not exist[/red]")
        sys.exit(1)

    # Process arguments into config
    config = {}
    if args.qualities:
        config['qualities'] = args.qualities
    if args.presets:
        config['presets'] = args.presets
    if args.dimensions:
        # Convert flat list to tuples of pairs
        if len(args.dimensions) % 2 != 0:
            console.print("[red]Error: Dimensions must be provided in pairs (width height)[/red]")
            sys.exit(1)
        config['target_dimensions'] = list(zip(args.dimensions[::2], args.dimensions[1::2]))
    if args.weights:
        config['weights'] = {
            'ssim': args.weights[0],
            'encode_time': args.weights[1],
            'file_size': args.weights[2]
        }
    if args.preserve_outputs:
        config['preserve_outputs'] = True
    if args.export_csv:
        config['export_csv'] = True

    # Show analysis header
    console.print(Panel.fit(
        f"[bold blue]Analyzing {'directory' if args.batch else 'image'}:[/bold blue] {args.input}",
        title="WebP Optimizer",
        border_style="blue"
    ))

    try:
        if args.batch:
            batch_optimizer = WebPBatchOptimizer(args.input)
            results = batch_optimizer.process_directory()
        else:
            optimizer = WebPOptimizer(args.input, config if config else None)
            results = optimizer.run_analysis()

        if not results:
            console.print("\n[red]No optimal configurations found. Please check the error messages above.[/red]")
            sys.exit(1)

        # Display results (existing code...)
        for result in results:
            # Create main configuration table
            table = Table(
                title=f"Optimal Configuration for {result['dimensions']}",
                show_header=True,
                header_style="bold magenta"
            )
            
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Dimensions", result['dimensions'])
            table.add_row("Best Preset", result['best_preset'])
            table.add_row("Quality", str(result['best_quality']))
            table.add_row("Encoding", "[green]Lossless[/green]" if result['is_lossless'] else f"[yellow]Lossy (q={result['best_quality']})[/yellow]")
            table.add_row("Encode Time", f"{result['encode_time']}s")
            table.add_row("File Size", f"{result['file_size_kb']}KB")
            table.add_row("SSIM Score", f"{result['ssim']}")
            table.add_row("PSNR Score", f"{result['psnr']}")
            if 'output_path' in result:
                table.add_row("Output Path", result['output_path'])
            
            console.print(table)
            
            # Create statistics table
            stats_table = Table(
                title="Performance Statistics",
                show_header=True,
                header_style="bold magenta"
            )
            
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Min", style="green")
            stats_table.add_column("Max", style="green")
            stats_table.add_column("Average", style="green")
            
            stats = result['stats']
            stats_table.add_row(
                "SSIM",
                str(stats['ssim']['min']),
                str(stats['ssim']['max']),
                str(stats['ssim']['avg'])
            )
            stats_table.add_row(
                "Encode Time (s)",
                str(stats['encode_time']['min']),
                str(stats['encode_time']['max']),
                str(stats['encode_time']['avg'])
            )
            stats_table.add_row(
                "File Size (KB)",
                str(stats['file_size']['min']),
                str(stats['file_size']['max']),
                str(stats['file_size']['avg'])
            )
            
            console.print(stats_table)
            console.print("\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
