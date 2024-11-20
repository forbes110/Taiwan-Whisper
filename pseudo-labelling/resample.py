import os
import glob
import argparse
import asyncio
import subprocess
import logging
import time
import shutil
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from asyncio import Queue
from typing import Set, List, Dict
from tqdm import tqdm
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TARGET_SAMPLE_RATE = 16000
UPDATE_INTERVAL = 0.1

class AsyncAudioProcessor:
    def __init__(self, max_workers: int = 4, output_dir: str = None):
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results_queue = Queue()
        self.stats: Dict[str, int] = {
            'processed': 0,
            'total': 0,
            'errors': 0
        }
        self.pbar = None

    async def load_invalid_channels(self, csv_path: str) -> Set[str]:
        if not csv_path:
            return set()

        try:
            def _read_csv():
                invalid_channels = set()
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        if row:
                            invalid_channels.add(row[0].strip())
                return invalid_channels

            invalid_channels = await asyncio.get_event_loop().run_in_executor(
                self.executor, _read_csv
            )
            logging.info(f"Loaded {len(invalid_channels)} invalid channels")
            return invalid_channels
        except Exception as e:
            logging.error(f"Error loading invalid channels file: {str(e)}")
            return set()

    def should_process_file(self, file_path: str, invalid_channels: Set[str]) -> bool:
        if not invalid_channels:
            return True
        path_parts = os.path.normpath(file_path).split(os.sep)
        return not any(part in invalid_channels for part in path_parts)

    async def resample_audio(self, input_path: str, relative_path: str, to_flac: bool = True) -> None:
        try:
            base, ext = os.path.splitext(os.path.basename(input_path))
            ext = ext.lower()
            # Determine output file path
            if self.output_dir:
                output_base_dir = os.path.join(self.output_dir, os.path.dirname(relative_path))
                os.makedirs(output_base_dir, exist_ok=True)
                output_filename = f"{base}.flac" if to_flac else f"{base}{ext}"
                output_path = os.path.join(output_base_dir, output_filename)
            else:
                output_path = f"{os.path.splitext(input_path)[0]}  c.flac" if to_flac else f"{os.path.splitext(input_path)[0]}  c{ext}"

            ffmpeg_command = [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", "1"
            ]

            if to_flac:
                ffmpeg_command.extend(["-c:a", "flac", "-compression_level", "8"])
            else:
                codec = "aac" if ext in ['.m4a', '.mp4'] else "pcm_s16le" if ext == '.wav' else "copy"
                ffmpeg_command.extend(["-c:a", codec])
            ffmpeg_command.append(output_path)

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()

            if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                if not self.output_dir:
                    def _handle_file_operations():
                        if to_flac and ext == ".flac":
                            shutil.move(output_path, input_path)
                        elif to_flac:
                            os.remove(input_path)
                            shutil.move(output_path, input_path.replace(ext, '.flac'))
                        else:
                            shutil.move(output_path, input_path)

                    await asyncio.get_event_loop().run_in_executor(self.executor, _handle_file_operations)
                await self.results_queue.put(("success", input_path))
            else:
                error_message = stderr.decode('utf-8') if stderr else "Unknown error"
                await self.results_queue.put(("error", f"{input_path}: {error_message}"))
                self.stats['errors'] += 1

        except Exception as e:
            await self.results_queue.put(("error", f"{input_path}: {str(e)}"))
            self.stats['errors'] += 1

        finally:
            self.stats['processed'] += 1
            if self.pbar:
                self.pbar.update(1)
                self.pbar.refresh()

    async def process_directory(self, input_path: str, invalid_channels: Set[str] = None, to_flac: bool = True) -> None:
        if invalid_channels is None:
            invalid_channels = set()

        semaphore = asyncio.Semaphore(self.max_workers)
        if os.path.isfile(input_path):
            relative_path = os.path.relpath(input_path, start=os.path.dirname(input_path))
            files = [ (input_path, relative_path) ] if self.should_process_file(input_path, invalid_channels) else []
        else:
            files = []
            for ext in ['.m4a', '.flac']:
                pattern = os.path.join(input_path, '**', f'*{ext}')
                for f in glob.glob(pattern, recursive=True):
                    if self.should_process_file(f, invalid_channels):
                        relative_path = os.path.relpath(f, start=input_path)
                        files.append( (f, relative_path) )

        if not files:
            logging.info("No valid audio files found")
            return

        self.stats['total'] = len(files)
        logging.info(f"Found {len(files)} valid audio files")

        self.pbar = tqdm(
            total=len(files),
            desc="Processing",
            unit="file",
            position=0,
            leave=True
        )

        async def process_with_semaphore(file_tuple):
            file_path, rel_path = file_tuple
            async with semaphore:
                await self.resample_audio(file_path, rel_path, to_flac)

        try:
            await asyncio.gather(
                *[process_with_semaphore(f) for f in files],
                return_exceptions=True
            )
        finally:
            if self.pbar:
                self.pbar.close()
                print("\n", end="")  # Force a newline after progress bar
                sys.stdout.flush()
                self.pbar = None

    async def cleanup(self):
        while not self.results_queue.empty():
            status, message = await self.results_queue.get()
            if status == "error":
                logging.error(message)
            else:
                logging.info(message)
        self.executor.shutdown(wait=True)
        if self.pbar:
            self.pbar.close()

async def main():
    parser = argparse.ArgumentParser(description="Async audio file processor")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--invalid_channels", type=str, help="Path to invalid channels CSV file")
    parser.add_argument("--to_flac", action='store_true', help="Convert output to FLAC format")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save processed files")
    args = parser.parse_args()

    start_time = time.time()
    processor = AsyncAudioProcessor(max_workers=args.max_workers, output_dir=args.output_dir)

    try:
        invalid_channels = await processor.load_invalid_channels(args.invalid_channels)
        await processor.process_directory(
            args.input,
            invalid_channels=invalid_channels,
            to_flac=args.to_flac
        )
        total_time = time.time() - start_time
        print(f"Completed in {total_time:.2f}s | "
              f"Processed: {processor.stats['processed']} | "
              f"Errors: {processor.stats['errors']}")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        await processor.cleanup()
        sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
